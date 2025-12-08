import copy
import itertools
import json
import pprint
import time
from collections import defaultdict
import os
from os import path as osp
from typing import Dict
from pathlib import Path


import numpy as np
import torch
import wandb
from loguru import logger
from scipy.stats import rankdata
from tqdm import tqdm

from zsceval.runner.shared.base_runner import *
from zsceval.runner.separated.overcooked_runner import OvercookedRunner
from zsceval.algorithms.hierarchical_marl_zsc.hmarl_trainer import HMARLTrainer, HMARLTrainer_PerAgent
from zsceval.algorithms.hierarchical_marl_zsc.hmarl_policy import HMARLModel
from zsceval.algorithms.hierarchical_marl_zsc.utils.replay_buffer import Replay_Buffer

from zsceval.utils.log_util import eta, get_table_str


def _t2n(x):
    return x.detach().cpu().numpy()


def _init_fcp_pool(pool_dir: Path):
    pool_dir.mkdir(parents=True, exist_ok=True)
    return sorted([p for p in pool_dir.glob("*.pt") if p.is_file()])


def _sample_fcp_pool(pool_paths, k: int):
    if k <= 0 or len(pool_paths) == 0:
        return []
    replace = len(pool_paths) < k
    sampled = np.random.choice(pool_paths, size=k, replace=replace)
    return list(sampled)

class OvercookedRunnerHMARL(OvercookedRunner):
    """
    Override some training method, buffers in OvercookedRunner to run HMARLModel, HMARLTrainer.
    """
    # override init method
    def __init__(self, config): 
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]
        if config.__contains__("render_envs"):
            self.render_envs = config["render_envs"]

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_single_network = self.all_args.use_single_network
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        # logging settings
        if self.use_render:
            # config["run_dir"] 는 Path 라고 가정
            self.run_dir = config["run_dir"]
            self.gif_dir = self.run_dir / "gifs"
            self.gif_dir.mkdir(parents=True, exist_ok=True)

            self.save_dir = self.run_dir / "models"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.use_wandb:
            # wandb.run.dir 은 문자열이므로 Path 로 감싸줌
            self.run_dir = Path(wandb.run.dir)
            self.save_dir = self.run_dir
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = self.run_dir / "logs"
            self.log_dir.mkdir(parents=True, exist_ok=True)

            self.writter = SummaryWriter(str(self.log_dir))

            self.save_dir = self.run_dir / "models"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.fcp_pool_dir = Path(getattr(self.all_args, "fcp_pool_dir", self.run_dir / "fcp_pool"))
        self.fcp_pool = _init_fcp_pool(self.fcp_pool_dir)
        self.fcp_partner_ids = []
        self.fcp_pool_add_step_threshold = getattr(
            self.all_args, "fcp_pool_add_step_threshold", self.num_env_steps
        )
        self.fcp_pool_min_eval_sparse = getattr(self.all_args, "fcp_pool_min_eval_sparse", -np.inf)
        self.last_eval_sparse = None

        # 나머지 코드에서 쓸 때는:
        # str(self.run_dir), str(self.save_dir) 로 필요할 때만 문자열 변환
        # load HMARL specific parameters from specific config directory
        trainer_cfg_path = self.all_args.hmarl_trainer_config_path
        cfg_namespace = {}
        with open(trainer_cfg_path, "r") as f:
            exec(f.read(), cfg_namespace)
        trainer_cfg = cfg_namespace["config"]["trainer"]
        model_cfg = cfg_namespace["config"]["model"]  # trainer config includes model config inside

        # =====================================================================
        # Override some parts of the PKL configs according to runtime all_args
        # =====================================================================
        # -----------------------
        # Override model config
        # -----------------------
        model_cfg["num_agents"] = self.num_agents
        # The actual observation/ shared observation/ action spaces come from env
        model_cfg["obs_space"] = self.envs.observation_space
        model_cfg["share_obs_space"] = (
            self.envs.share_observation_space
            if self.use_centralized_V
            else self.envs.observation_space
        )
        model_cfg["act_space"] = self.envs.action_space
        # Whether we use obs instead of state (for centralized critic)
        model_cfg["use_obs_instead_of_state"] = self.use_obs_instead_of_state

        # These come from env action space (not PKL)
        model_cfg["num_actions"] = self.envs.action_space[0].n
        model_cfg["obs_channels"] = self.envs.observation_space[0].shape[-1]
        model_cfg["share_obs_channels"] = self.envs.share_observation_space[0].shape[-1]
        model_cfg["obs_height"] = self.envs.observation_space[0].shape[0]
        model_cfg["obs_width"] = self.envs.observation_space[0].shape[1]
        # -----------------------
        # Override trainer config
        # -----------------------
        # HMARLTrainer’s batch_size = number of parallel rollout envs
        trainer_cfg["batch_size"] = self.all_args.n_rollout_threads
        # Number of skills in trainer must match model
        trainer_cfg["N_skills"] = model_cfg["num_skills"]
        # Ensure steps_per_assign is mirrored
        trainer_cfg["steps_per_assign"] = model_cfg["steps_per_assign"]
        

        # Create instance of algorithm per agent (same config, separate parameters)
        TrainAlgo, Policy = HMARLTrainer_PerAgent, HMARLModel
        combined_cfg = {"trainer": trainer_cfg, "model": model_cfg}
        self.trainer = []
        self.policy = []

        for _ in range(self.num_agents):
            cfg_copy = copy.deepcopy(combined_cfg)
            trainer = TrainAlgo(cfg_copy, self.device)
            self.trainer.append(trainer)
            self.policy.append(trainer.hsd)

        # dump policy config to allow loading population in yaml form
        self.policy_config = copy.deepcopy(model_cfg)
        policy_config_path = os.path.join(self.run_dir, "policy_config.pkl")
        pickle.dump(self.policy_config, open(policy_config_path, "wb"))
        print(f"Pickle dump policy config at {policy_config_path}")
        
        # not implemented?
        if "store" in self.experiment_name:
            exit()

        # load pretrained policy
        if self.model_dir is not None: 
            self.restore() 

        # for training br
        self.br_best_sparse_r = 0
        self.br_eval_json = {}

    def _load_fcp_partners(self):
        self.fcp_partner_ids = []
        num_partners = max(0, min(self.num_agents - 1, len(self.trainer) - 1))
        sampled_paths = _sample_fcp_pool(self.fcp_pool, num_partners)
        if not sampled_paths:
            return
        for idx, model_path in enumerate(sampled_paths, start=1): # 0 stands for trainable ego
            self.trainer[idx].hsd.load(str(model_path), map_location=self.device)
            self.policy[idx] = self.trainer[idx].hsd
            self.fcp_partner_ids.append(idx)

    def _append_agent0_to_pool(self, total_num_steps: int):
        if not self.trainer:
            return
        if total_num_steps < self.fcp_pool_add_step_threshold:
            logger.info(
                f"[FCP pool] skip append: total steps {total_num_steps} < threshold {self.fcp_pool_add_step_threshold}"
            )
            return
        if self.fcp_pool_min_eval_sparse > -np.inf:
            if self.last_eval_sparse is None or self.last_eval_sparse < self.fcp_pool_min_eval_sparse:
                logger.info(
                    f"[FCP pool] skip append: eval sparse {self.last_eval_sparse} < min "
                    f"{self.fcp_pool_min_eval_sparse}"
                )
                return
        timestamp = int(time.time() * 1000)
        save_path = self.fcp_pool_dir / f"pool_agent0_{timestamp}.pt"
        self.trainer[0].hsd.save(str(save_path))
        self.fcp_pool.append(save_path)

    def run_stage(self): # uploads nontrainable trainer&policies to agent index 1 ~ num_agent-1 
        # train sp
        self._load_fcp_partners()
        obs, share_obs, available_actions = self.warmup() # changed from original warmup

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0

        for episode in range(episodes):
            s_time = time.time()

            for step in range(self.episode_length):
                # Sample actions based on recent environment interaction
                # Trainer internally hides actual details, only prints out low level action
                actions = self.collect(step, obs, share_obs, available_actions) # [n_rollout_threads, num_agents,] where each entry is action 0 ~ 5 
                # Interact with the environment to get observations, rewards, and next observations
                (
                    _obs_batch_single_agent,
                    share_obs_next,
                    rewards, # shape: [n_rollout_threads, num_agents (actual), 1]
                    dones,
                    infos,
                    available_actions_next,
                ) = self.envs.step(actions) # actions requirement: [n_rollout_threads, num_agents, 1] 0 ~ 5

                # ===>>> Extract all_agent_obs from info_list <<<===
                obs_next, share_obs_next, available_actions_next = self.info_translation(infos)
                total_num_steps += self.n_rollout_threads
                self.envs.anneal_reward_shaping_factor([total_num_steps] * self.n_rollout_threads)

                data = (
                    step, obs, share_obs, actions, rewards, obs_next, share_obs_next, dones
                )
                self.insert(data)

                obs, share_obs, rewards, available_actions = obs_next, share_obs_next, rewards, available_actions_next

            print("episode training finished: ", episode)
            e_time = time.time()
            logger.trace(f"Rollout time: {e_time - s_time:.3f}s")

            # compute return and update network
            s_time = time.time()
            # self.compute()
            train_infos = self.train(episode)
            e_time = time.time()
            logger.trace(f"Update models time: {e_time - s_time:.3f}s")

            # post process
            s_time = time.time()
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model (overriden because HMARL has multiple networks than rMAPPO, ...)
            if episode < 50:
                if episode % 2 == 0:
                    # self.save(episode)
                    self.save(total_num_steps)
            elif episode < 100:
                if episode % 5 == 0:
                    self.save(total_num_steps)
                    # self.save(episode)
            else:
                if episode % self.save_interval == 0 or episode == episodes - 1:
                    self.save(total_num_steps)
                    # self.save(episode)
            
            print("episode training finished: ", episode)
            # log information
            if episode % self.log_interval == 0 or episode == episodes - 1:
                end = time.time()
                eta_t = eta(start, end, self.num_env_steps, total_num_steps)
                log_data = list(
                    {
                        "Layout": self.all_args.layout_name,
                        "Algorithm": self.algorithm_name,
                        "Experiment": self.experiment_name,
                        "Seed": self.all_args.seed,
                        "Episodes": episode,
                        "Total Episodes": episodes,
                        "Timesteps": total_num_steps,
                        "Total Timesteps": self.num_env_steps,
                        "FPS": int(total_num_steps / (end - start)),
                        "ETA": eta_t,
                    }.items()
                )
                logger.info("training process:\n" + get_table_str(log_data))
                logger.info(
                    "Layout {} Algo {} Exp {} Seed {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}, ETA {}.".format(
                        self.all_args.layout_name,
                        self.algorithm_name,
                        self.experiment_name,
                        self.all_args.seed,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                        eta_t,
                    )
                )

                # HMARL-specific debug logs
                for agent_id in range(self.num_agents):
                    dep_key = f"decoder_expected_prob_agent{agent_id}"
                    if dep_key in train_infos:
                        logger.info(f"[agent{agent_id}] decoder expected prob={train_infos[dep_key]:.4f}")

                for agent_id in range(self.num_agents):
                    ir_key = f"intrinsic_reward_mean_agent{agent_id}"
                    if ir_key in train_infos:
                        logger.info(f"[agent{agent_id}] intrinsic_reward_mean={train_infos[ir_key]:.4f}")

                for agent_id in range(self.num_agents):
                    eps_key = f"epsilon_agent{agent_id}"
                    alpha_key = f"alpha_agent{agent_id}"
                    if eps_key in train_infos and alpha_key in train_infos:
                        logger.info(
                            f"[agent{agent_id}] epsilon={train_infos[eps_key]:.4f}, alpha={train_infos[alpha_key]:.4f}"
                        )

                for agent_id in range(self.num_agents):
                    hl_key = f"high_level_reward_mean_agent{agent_id}"
                    if hl_key in train_infos:
                        logger.info(f"[agent{agent_id}] high_level_reward_mean={train_infos[hl_key]:.4f}")

                for agent_id in range(self.num_agents):
                    su_key = f"skill_usage_agent{agent_id}"
                    if su_key in train_infos:
                        su = train_infos[su_key]
                        logger.info(f"[agent{agent_id}] skill_usage={['{:.2f}'.format(x) for x in su]}")

                # # shaped reward
                # train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                # logger.info("average episode rewards is {:.3f}".format(train_infos["average_episode_rewards"]))

                # get information of env
                env_infos = defaultdict(list)
                if self.env_name == "Overcooked":
                    if self.all_args.overcooked_version == "old":
                        from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
                            SHAPED_INFOS,
                        )
                        shaped_info_keys = SHAPED_INFOS
                    else:
                        from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
                            SHAPED_INFOS,
                        )
                        shaped_info_keys = SHAPED_INFOS
                    for info in infos:
                        for a in range(self.num_agents):
                            env_infos[f"ep_sparse_r_by_agent{a}"].append(info["episode"]["ep_sparse_r_by_agent"][a])
                            env_infos[f"ep_shaped_r_by_agent{a}"].append(info["episode"]["ep_shaped_r_by_agent"][a])
                            
                            for i, k in enumerate(shaped_info_keys):
                                env_infos[f"ep_{k}_by_agent{a}"].append(info["episode"]["ep_category_r_by_agent"][a][i])
                        env_infos["ep_sparse_r"].append(info["episode"]["ep_sparse_r"])
                        env_infos["ep_shaped_r"].append(info["episode"]["ep_shaped_r"])
                print("train_infos:", train_infos)
                self.log_train(train_infos, total_num_steps) # requirement: train_info be [num_agents] with scalar values
                self.log_env(env_infos, total_num_steps) # prints log env infos
                if self.use_wandb:
                    wandb.log({"train/ETA": eta_t}, step=total_num_steps)
                logger.info(f'average sparse rewards is {np.mean(env_infos["ep_sparse_r"]):.3f}')
            
            # eval
            if episode % self.eval_interval == 0 and self.use_eval or episode == episodes - 1:
                self.eval(total_num_steps)
            e_time = time.time()
            logger.trace(f"Post update models time: {e_time - s_time:.3f}s")
        self._append_agent0_to_pool(total_num_steps)

    def warmup(self): # override to fit HMARLTrainer
        # ===>>> 표준 Gym format: reset returns (obs_batch, info_list) <<<===
        obs_batch, info_list = self.envs.reset()
        
        # ===>>> info_list에서 데이터 추출 <<<===
        # all_agent_obs 추출 및 쌓기 -> (n_rollout_threads, num_agents, H, W, C)
        all_agent_obs = np.array([info['all_agent_obs'] for info in info_list])
        
        # share_obs 추출 및 쌓기 -> (n_rollout_threads, num_agents, H, W, C_share)
        share_obs = np.array([info['share_obs'] for info in info_list])
        
        # available_actions 추출 및 쌓기 -> (n_rollout_threads, num_agents, num_actions)
        available_actions = np.array([info['available_actions'] for info in info_list])
        # ===>>> 추출 끝 <<<===

        # 우리가 쓰는 버퍼는 오로지 Q함수 학습용이므로 warmup 함수를 변화시킨다.
        return all_agent_obs, share_obs, available_actions

    # # merge rollout, batch into single batch dimension (also done in overcooked_runner)
    # def transform(self, input):
    #     shape = input.shape
    #     return input.reshape(shape[0] * shape[1], *shape[2:])
    # # separete batch dim into rollout, batch dims (rollout comes first)
    # def inverse_transform(self, input, n_rollout_threads):
    #     shape = input.shape
    #     return input.reshape(n_rollout_threads, shape[0] // n_rollout_threads, *shape[1:])

    def info_translation(self, info_list):
        # Extract data from info_list
        all_agent_obs = np.array([info['all_agent_obs'] for info in info_list])
        share_obs = np.array([info['share_obs'] for info in info_list])
        available_actions = np.array([info['available_actions'] for info in info_list])
        return all_agent_obs, share_obs, available_actions


    # from current step inside episode, collect actions and related variables for trainer
    # only used during training step
    @torch.no_grad()
    def collect(self, step, obs, share_obs, available_actions): # override to fit HMARLTrainer
        actions_by_agent = []
        for agent_id, trainer in enumerate(self.trainer):
            trainer.prep_rollout() # set eval mode for policy
            obs = obs[:,agent_id] # [n_rollout_threads, H, W, C]
            share_obs = share_obs[:,agent_id] # [n_rollout_threads, H, W, C_share]
            available_actions = available_actions[:,agent_id] # [n_rollout_threads, num_actions]
            # trainer internally includes policy, and updates buffers, current skills, intrinsic rewards, ... internally
            # so it only prints out actions, actual training algorithm of hmarl is hidden
            agent_actions = trainer.get_actions_algorithm(step, obs, share_obs, available_actions)
            agent_actions = np.asarray(agent_actions) # [n_rollout_threads, 1, 1]
            agent_actions = agent_actions.squeeze(1) # remove the agent dimension
            actions_by_agent.append(agent_actions)

        stacked_actions = np.stack(actions_by_agent, axis=1)
        return stacked_actions # [n_rollout_threads, num_agents, 1]

    def insert(self, data): # override to fit HMARLTrainer
        step, obs, share_obs, actions, rewards, obs_next, share_obs_next, dones = data

        for agent_id, trainer in enumerate(self.trainer):
            trainer.update_buffer(
                step,
                obs[:,agent_id], # [n_rollout_threads, H, W, C]
                share_obs[:,agent_id], # [n_rollout_threads, H, W, C_share]
                actions[:,agent_id], # [n_rollout_threads, 1]
                rewards,
                obs_next,
                share_obs_next,
                dones,
            )


    def restore(self):
        for agent_id, trainer in enumerate(self.trainer):
            agent_model_dir = getattr(self.all_args, f"model_dir_agent{agent_id}", None) or self.model_dir
            if agent_model_dir is None:
                continue

            policy_cfg_path = os.path.join(agent_model_dir, "policy_config.pkl")
            if os.path.exists(policy_cfg_path):
                model_cfg = pickle.load(open(policy_cfg_path, "rb"))
                trainer.hsd = HMARLModel(model_cfg, device=self.device)
                self.policy[agent_id] = trainer.hsd

            model_files = [f for f in os.listdir(agent_model_dir) if f.startswith("model_")]
            if not model_files:
                continue
            model_files.sort()
            latest_model = model_files[-1]

            model_path = os.path.join(agent_model_dir, latest_model)
            trainer.hsd.load(model_path, map_location=self.device)


    def train(self, num_steps: int = 0):
        all_train_infos = {}
        for agent_id, trainer in enumerate(self.trainer):
            if agent_id in self.fcp_partner_ids:
                continue
            trainer.prep_training()
            train_info = trainer.training_step(num_steps)
            for k, v in train_info.items():
                all_train_infos[f"{k}_agent{agent_id}"] = v
        print("train_infos in overcooked_runner_hmarl:", all_train_infos)
        return all_train_infos # return dict, not list
    
    def save(self, step): # override to store all networks of HMARL (TODO)
        for agent_id, trainer in enumerate(self.trainer):
            agent_save_dir = Path(self.save_dir) / f"agent{agent_id}"
            trainer.save(step, str(agent_save_dir))

    # change eval to fit HMARLTrainer
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_env_infos = defaultdict(list)
        
        # shaped info keys
        if self.env_name == "Overcooked":
            if self.all_args.overcooked_version == "old":
                from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS
            else:
                from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS
            shaped_info_keys = SHAPED_INFOS

        # === Reset eval envs ===
        obs_batch, info_list = self.eval_envs.reset()
        obs = np.array([info['all_agent_obs'] for info in info_list])
        share_obs = np.array([info['share_obs'] for info in info_list])
        available_actions = np.array([info['available_actions'] for info in info_list])

        # For logging rewards
        episode_rewards = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        for trainer in self.trainer:
            trainer.prep_rollout()  # set eval mode

        for step in range(self.episode_length):
            actions_by_agent = []
            for agent_id, trainer in enumerate(self.trainer):
                agent_actions = trainer.hsd.get_actions_algorithm(
                    step,
                    obs,
                    share_obs,
                    available_actions,
                    epsilon=0.0  # no exploration in eval
                )
                agent_actions = np.asarray(agent_actions)
                if agent_actions.ndim == 3 and agent_actions.shape[-1] == 1:
                    agent_actions = agent_actions.squeeze(-1)
                actions_by_agent.append(agent_actions[:, agent_id])

            actions = np.expand_dims(np.stack(actions_by_agent, axis=1), axis=-1)
            # Step the environment
            (
                _obs_single_agent,
                share_obs_next,
                rewards,
                dones,
                infos,
                available_actions_next,
            ) = self.eval_envs.step(actions) # actions requirement: [n_eval_rollout_threads, num_agents,]

            # Extract next obs
            obs_next, share_obs_next, available_actions_next = self.info_translation(infos)

            # Accumulate rewards
            episode_rewards += rewards.squeeze(-1)

            obs, share_obs, available_actions = obs_next, share_obs_next, available_actions_next

        # --- Logging ---
        for eval_info in infos:
            ep = eval_info["episode"]
            eval_env_infos["eval_sparse_r"].append(ep["ep_sparse_r"])
            eval_env_infos["eval_shaped_r"].append(ep["ep_shaped_r"])

        eval_env_infos["eval_average_episode_rewards"] = np.mean(eval_env_infos["eval_shaped_r"])
        self.log_env(eval_env_infos, total_num_steps)

        eval_env_infos["eval_average_episode_rewards"] = np.mean(episode_rewards)

        self.log_env(eval_env_infos, total_num_steps)
        if eval_env_infos["eval_sparse_r"]:
            self.last_eval_sparse = float(np.mean(eval_env_infos["eval_sparse_r"]))

    @torch.no_grad()
    def render(self):
        envs = self.envs

        obs_batch, info_list = envs.reset()
        obs = np.array([info['all_agent_obs'] for info in info_list])
        share_obs = np.array([info['share_obs'] for info in info_list])
        available_actions = np.array([info['available_actions'] for info in info_list])

        for episode in tqdm(range(self.all_args.render_episodes)):
            episode_rewards = np.zeros((self.n_render_rollout_threads, self.num_agents))

            for step in range(self.episode_length):
                for trainer in self.trainer:
                    trainer.prep_rollout()

                actions_by_agent = []
                for agent_id, trainer in enumerate(self.trainer):
                    agent_actions = trainer.hsd.get_actions_algorithm(
                        step,
                        obs,
                        share_obs,
                        available_actions,
                        epsilon=0.0,
                    )
                    agent_actions = np.asarray(agent_actions)
                    if agent_actions.ndim == 3 and agent_actions.shape[-1] == 1:
                        agent_actions = agent_actions.squeeze(-1)
                    actions_by_agent.append(agent_actions[:, agent_id])

                actions = np.expand_dims(np.stack(actions_by_agent, axis=1), axis=-1)
                print("render actions shape:", actions.shape)
                (
                    _obs_single_agent,
                    share_obs_next,
                    rewards,
                    dones,
                    infos,
                    available_actions_next,
                ) = envs.step(actions)

                obs_next, share_obs_next, available_actions_next = self.info_translation(infos)

                episode_rewards += rewards

                obs, share_obs, available_actions = obs_next, share_obs_next, available_actions_next

            logger.info("render average episode rewards: "
                        f"{np.mean(np.sum(episode_rewards, axis=1)):.3f}")

    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            if isinstance(v, Iterable) and not isinstance(v, str):
                if len(v) == 0:
                    continue
                v = np.mean(v)

            log_key = f"train/{k}"

            if self.use_wandb:
                wandb.log({log_key: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(log_key, {log_key: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if isinstance(v, Iterable) and not isinstance(v, str):
                if len(v) == 0:
                    continue
                v = np.mean(v)

            log_key = f"env/{k}"

            if self.use_wandb:
                wandb.log({log_key: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(log_key, {log_key: v}, total_num_steps)
