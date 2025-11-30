import copy
import itertools
import json
import pprint
import time
from collections import defaultdict
from os import path as osp
from typing import Dict

import numpy as np
import torch
import wandb
from loguru import logger
from scipy.stats import rankdata
from tqdm import tqdm

from zsceval.runner.shared.base_runner import *
from zsceval.runner.separated.overcooked_runner import OvercookedRunner
from zsceval.algorithms.hierarchical_marl_zsc.hmarl_trainer import HMARLTrainer
from zsceval.algorithms.hierarchical_marl_zsc.hmarl_policy import HMARLModel
from zsceval.algorithms.hierarchical_marl_zsc.utils.replay_buffer import Replay_Buffer

from zsceval.utils.log_util import eta, get_table_str


def _t2n(x):
    return x.detach().cpu().numpy()


class OvercookedRunnerHMARL(OvercookedRunner):
    """
    Override some training method, buffers in OvercookedRunner to run HMARLModel, HMARLTrainer.
    """

    def __init__(self, config): # override init method because buffer, trainer, policy are different
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

        if self.use_render:
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / "gifs")
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / "logs")
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / "models")
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
        
        # log env info
        logger.info(
            f"Action space {self.envs.action_space[0]}, Obs space {self.envs.observation_space[0].shape}, Share obs space {share_observation_space.shape}"
        )
        share_observation_space = (
            self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        )

        # Create instance of algorithm (TODO)
        TrainAlgo, Policy = HMARLTrainer, HMARLModel

        # policy network
        self.policy = Policy(
            # state args here TODO
        )

        # dump policy config to allow loading population in yaml form
        self.policy_config = (
            # copy above parameters used to initialize Policy class except device TODO
            # additional important parameters would be high skill period (because decoder depends on it)
        )
        policy_config_path = os.path.join(self.run_dir, "policy_config.pkl")
        pickle.dump(self.policy_config, open(policy_config_path, "wb"))
        print(f"Pickle dump policy config at {policy_config_path}")
        if "store" in self.experiment_name:
            exit()

        if self.model_dir is not None:
            self.restore() # Reimplement this to fit Policy (TODO)

        # trainer - takes policy instance (but not environment instance or buffer)
        self.trainer = TrainAlgo(
            # pass necessary args including self.policy TODO
        )

        # for training br
        self.br_best_sparse_r = 0
        self.br_eval_json = {}

    def run(self):
        # train sp
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
                    rewards,
                    dones,
                    infos,
                    available_actions_next,
                ) = self.envs.step(actions)

                # ===>>> Extract all_agent_obs from info_list <<<===
                obs_next = np.array([info['all_agent_obs'] for info in infos])
                total_num_steps += self.n_rollout_threads
                self.envs.anneal_reward_shaping_factor([total_num_steps] * self.n_rollout_threads)

                self.trainer.update_buffer(
                    step,
                    obs,
                    share_obs,
                    actions,
                    rewards,
                    obs_next,
                    share_obs_next,
                    dones,
                )

                obs, share_obs, rewards, available_actions = obs_next, share_obs_next, rewards, available_actions_next


            e_time = time.time()
            logger.trace(f"Rollout time: {e_time - s_time:.3f}s")

            # compute return and update network
            s_time = time.time()
            self.compute()
            train_infos = self.train(total_num_steps)
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
                # shaped reward
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                logger.info("average episode rewards is {:.3f}".format(train_infos["average_episode_rewards"]))

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

    # from current step inside episode, collect actions and related variables for trainer
    # only used during training step
    @torch.no_grad()
    def collect(self, step, obs, share_obs, available_actions): # override to fit HMARLTrainer
        self.trainer.prep_rollout() # set eval mode for policy

        # trainer internally includes policy, and updates buffers, current skills, intrinsic rewards, ... internally
        # so it only prints out actions, actual training algorithm of hmarl is hidden
        actions = self.trainer.get_actions_trainmode(step, obs, share_obs, available_actions)

        return actions

    def restore(self): # only called when loading pretrained policy at init
        policy_model_state_dict = torch.load(str(self.model_dir) + "/model.pt", map_location=self.device)
        self.policy.model.load_state_dict(policy_model_state_dict)


    def train(self, num_steps: int = 0): # override to fit HMARLTrainer
        train_infos = [] # list of dict for each agent where it prints log info
        self.trainer.prep_training() # ensure to change policy models to training mode
        train_info = self.trainer.training_step(num_steps) # internally trains high low Q and decoder according to num_steps and returns log info
        train_infos.append(train_info)
        # self.log_system() not implemented even in original runners
        return train_infos
    
    def save(self, step): # override to store all networks of HMARL (TODO)
        # logger.info(f"save sp periodic_{step}.pt")
        # if self.use_single_network:
        #     policy_model = self.trainer.policy.model
        #     torch.save(
        #         policy_model.state_dict(),
        #         str(self.save_dir) + f"/model_periodic_{step}.pt",
        #     )
        # else:
        #     policy_actor = self.trainer.policy.actor
        #     torch.save(
        #         policy_actor.state_dict(),
        #         str(self.save_dir) + f"/actor_periodic_{step}.pt",
        #     )
        #     if save_critic:
        #         policy_critic = self.trainer.policy.critic
        #         torch.save(
        #             policy_critic.state_dict(),
        #             str(self.save_dir) + f"/critic_periodic_{step}.pt",
        #         )
        self.trainer.save(step, self.save_dir)

    # change eval to fit HMARLTrainer
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_env_infos = defaultdict(list)
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
        eval_average_episode_rewards = []

        # ===>>> 표준 Gym format: reset returns (obs_batch, info_list) <<<===
        eval_obs_batch, eval_info_list = self.eval_envs.reset()

        # Extract all_agent_obs, shared_obs and available_actions from info_list
        eval_obs = np.array([info['all_agent_obs'] for info in eval_info_list])
        shared_obs = np.array([info['share_obs'] for info in eval_info_list])
        eval_available_actions = np.array([info['available_actions'] for info in eval_info_list])

        # Run policy's get_action where policy internally manages skill selection and low level action selection
        self.trainer.prep_rollout() # set eval mode for policy

        for steps in range(self.episode_length):
            eval_action = self.trainer.policy.act_in_pretrained(
                steps,
                np.concatenate(eval_obs),
                np.concatenate(eval_available_actions)
            )
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            
            # Obser reward and next obs
            (
                _eval_obs_batch_single_agent,
                _,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            # ===>>> Extract all_agent_obs from info_list <<<===
            eval_obs = np.array([info['all_agent_obs'] for info in eval_infos])
            eval_average_episode_rewards.append(eval_rewards)

        # logging
        for eval_info in eval_infos:
            for a in range(self.num_agents):
                eval_env_infos[f"eval_ep_sparse_r_by_agent{a}"].append(eval_info["episode"]["ep_sparse_r_by_agent"][a])
                eval_env_infos[f"eval_ep_shaped_r_by_agent{a}"].append(eval_info["episode"]["ep_shaped_r_by_agent"][a])
                for i, k in enumerate(shaped_info_keys):
                    eval_env_infos[f"eval_ep_{k}_by_agent{a}"].append(
                        eval_info["episode"]["ep_category_r_by_agent"][a][i]
                    )
            eval_env_infos["eval_ep_sparse_r"].append(eval_info["episode"]["ep_sparse_r"])
            eval_env_infos["eval_ep_shaped_r"].append(eval_info["episode"]["ep_shaped_r"])

        eval_env_infos["eval_average_episode_rewards"] = np.sum(eval_average_episode_rewards, axis=0)
        logger.success(
            f'eval average sparse rewards {np.mean(eval_env_infos["eval_ep_sparse_r"]):.3f} {len(eval_env_infos["eval_ep_sparse_r"])} episodes, total num timesteps {total_num_steps}/{self.num_env_steps}'
        )
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self): # logs average rewards
        envs = self.envs
        # ===>>> 표준 Gym format: reset returns (obs_batch, info_list) <<<===
        obs_batch, info_list = envs.reset()
        # Extract all_agent_obs and available_actions from info_list
        obs = np.array([info['all_agent_obs'] for info in info_list])
        available_actions = np.array([info['available_actions'] for info in info_list])

        for episode in tqdm(range(self.all_args.render_episodes)):
            rnn_states = np.zeros( # print skills instead of rnn_states (TODO)
                (self.n_rollout_threads, *self.buffer.rnn_states.shape[2:]),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []
            for step in range(self.episode_length):
                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act_in_pretrained(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    np.concatenate(available_actions),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
                # Obser reward and next obs
                _obs_batch_single_agent, _, rewards, dones, infos, available_actions = envs.step(actions)
                # ===>>> Extract all_agent_obs from info_list <<<===
                obs = np.array([info['all_agent_obs'] for info in infos])

                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

            logger.info("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))
