import copy
import itertools
import json
import pprint
import time
from collections import defaultdict
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
from zsceval.algorithms.hierarchical_marl_zsc.hmarl_trainer_mng import ManagerTrainerSingle as HMARLTrainer
from zsceval.algorithms.hierarchical_marl_zsc.hmarl_policy import HMARLModel
from zsceval.algorithms.hierarchical_marl_zsc.utils.replay_buffer import Replay_Buffer

from zsceval.utils.log_util import eta, get_table_str


def _t2n(x):
    return x.detach().cpu().numpy()


class OvercookedRunnerHMARL_mng(OvercookedRunner):
    """
    Same as OvercookedRunnerHMARL, but uses manager-based trainer (hmarl_trainer_mng.py).
    """

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
            self.run_dir = config["run_dir"]
            self.gif_dir = self.run_dir / "gifs"
            self.gif_dir.mkdir(parents=True, exist_ok=True)

            self.save_dir = self.run_dir / "models"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.use_wandb:
            self.run_dir = Path(wandb.run.dir)
            self.save_dir = self.run_dir
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = self.run_dir / "logs"
            self.log_dir.mkdir(parents=True, exist_ok=True)

            self.writter = SummaryWriter(str(self.log_dir))

            self.save_dir = self.run_dir / "models"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # load HMARL specific parameters from config file
        trainer_cfg_path = self.all_args.hmarl_trainer_config_path
        cfg_namespace = {}
        with open(trainer_cfg_path, "r") as f:
            exec(f.read(), cfg_namespace)
        trainer_cfg = cfg_namespace["config"]["trainer"]
        model_cfg = cfg_namespace["config"]["model"]

        # Override model config with env-provided shapes
        model_cfg["num_agents"] = self.num_agents
        model_cfg["obs_space"] = self.envs.observation_space
        model_cfg["share_obs_space"] = (
            self.envs.share_observation_space
            if self.use_centralized_V
            else self.envs.observation_space
        )
        model_cfg["act_space"] = self.envs.action_space
        model_cfg["use_obs_instead_of_state"] = self.use_obs_instead_of_state
        model_cfg["num_actions"] = self.envs.action_space[0].n
        model_cfg["obs_channels"] = self.envs.observation_space[0].shape[-1]
        model_cfg["share_obs_channels"] = self.envs.share_observation_space[0].shape[-1]
        model_cfg["obs_height"] = self.envs.observation_space[0].shape[0]
        model_cfg["obs_width"] = self.envs.observation_space[0].shape[1]

        # Override trainer config
        trainer_cfg["batch_size"] = self.all_args.n_rollout_threads
        trainer_cfg["N_skills"] = model_cfg["num_skills"]
        trainer_cfg["steps_per_assign"] = model_cfg["steps_per_assign"]

        # Instantiate manager trainer and policy
        TrainAlgo, Policy = HMARLTrainer, HMARLModel
        combined_cfg = {"trainer": trainer_cfg, "model": model_cfg}
        self.trainer = TrainAlgo(combined_cfg, self.device)
        self.policy = self.trainer.hsd

        # dump policy config to allow loading population in yaml form
        self.policy_config = model_cfg
        policy_config_path = os.path.join(self.run_dir, "policy_config.pkl")
        pickle.dump(self.policy_config, open(policy_config_path, "wb"))
        print(f"Pickle dump policy config at {policy_config_path}")

        if "store" in self.experiment_name:
            exit()

        if self.model_dir is not None:
            self.restore()

        self.br_best_sparse_r = 0
        self.br_eval_json = {}

    def run(self):
        obs, share_obs, available_actions = self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0

        for episode in range(episodes):
            s_time = time.time()

            for step in range(self.episode_length):
                actions = self.collect(step, obs, share_obs, available_actions)
                (
                    _obs_batch_single_agent,
                    share_obs_next,
                    rewards,
                    dones,
                    infos,
                    available_actions_next,
                ) = self.envs.step(actions)

                obs_next, share_obs_next, available_actions_next = self.info_translation(infos)
                total_num_steps += self.n_rollout_threads
                self.envs.anneal_reward_shaping_factor([total_num_steps] * self.n_rollout_threads)

                data = (
                    step, obs, share_obs, actions, rewards, obs_next, share_obs_next, dones
                )
                self.insert(data)

                obs, share_obs, rewards, available_actions = obs_next, share_obs_next, rewards, available_actions_next

            e_time = time.time()
            logger.trace(f"Rollout time: {e_time - s_time:.3f}s")

            s_time = time.time()
            train_infos = self.train(episode)
            e_time = time.time()
            logger.trace(f"Update models time: {e_time - s_time:.3f}s")

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode < 50:
                if episode % 2 == 0:
                    self.save(total_num_steps)
            elif episode < 100:
                if episode % 5 == 0:
                    self.save(total_num_steps)
            else:
                if episode % self.save_interval == 0 or episode == episodes - 1:
                    self.save(total_num_steps)

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

                if "epsilon" in train_infos:
                    logger.info(f"epsilon={train_infos['epsilon']:.4f}, alpha={train_infos['alpha']:.4f}")
                if "decoder_expected_prob" in train_infos:
                    logger.info(f"decoder expected prob={train_infos['decoder_expected_prob']:.4f}")
                if "intrinsic_reward_mean" in train_infos:
                    logger.info(f"intrinsic_reward_mean={train_infos['intrinsic_reward_mean']:.4f}")
                if "high_level_reward_mean" in train_infos:
                    logger.info(f"high_level_reward_mean={train_infos['high_level_reward_mean']:.4f}")
                if "skill_usage" in train_infos:
                    su = train_infos["skill_usage"]
                    logger.info(f"skill_usage={['{:.2f}'.format(x) for x in su]}")

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
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                if self.use_wandb:
                    wandb.log({"train/ETA": eta_t}, step=total_num_steps)
                print(env_infos["ep_sparse_r"])
                logger.info(f'average sparse rewards is {np.mean(env_infos["ep_sparse_r"]):.3f}')

            if episode % self.eval_interval == 0 and self.use_eval or episode == episodes - 1:
                self.eval(total_num_steps)

    def warmup(self):
        obs_batch, info_list = self.envs.reset()
        all_agent_obs = np.array([info['all_agent_obs'] for info in info_list])
        share_obs = np.array([info['share_obs'] for info in info_list])
        available_actions = np.array([info['available_actions'] for info in info_list])
        return all_agent_obs, share_obs, available_actions

    def info_translation(self, info_list):
        all_agent_obs = np.array([info['all_agent_obs'] for info in info_list])
        share_obs = np.array([info['share_obs'] for info in info_list])
        available_actions = np.array([info['available_actions'] for info in info_list])
        return all_agent_obs, share_obs, available_actions

    @torch.no_grad()
    def collect(self, step, obs, share_obs, available_actions):
        self.trainer.prep_rollout()
        actions = self.trainer.get_actions_algorithm(step, obs, share_obs, available_actions)
        return actions

    def insert(self, data):
        step, obs, share_obs, actions, rewards, obs_next, share_obs_next, dones = data
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

    def restore(self):
        policy_cfg_path = os.path.join(self.model_dir, "policy_config.pkl")
        model_cfg = pickle.load(open(policy_cfg_path, "rb"))
        self.policy = HMARLModel(model_cfg)

        model_files = [f for f in os.listdir(self.model_dir) if f.startswith("model_")]
        model_files.sort()
        latest_model = model_files[-1]

        model_path = os.path.join(self.model_dir, latest_model)
        self.policy.load(model_path)

    def train(self, num_steps: int = 0):
        self.trainer.prep_training()
        train_infos = self.trainer.training_step(num_steps)
        return train_infos

    def save(self, step):
        self.trainer.save(step, self.save_dir)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_env_infos = defaultdict(list)
        if self.env_name == "Overcooked":
            if self.all_args.overcooked_version == "old":
                from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS
            else:
                from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import SHAPED_INFOS
            shaped_info_keys = SHAPED_INFOS

        obs_batch, info_list = self.eval_envs.reset()
        obs = np.array([info['all_agent_obs'] for info in info_list])
        share_obs = np.array([info['share_obs'] for info in info_list])
        available_actions = np.array([info['available_actions'] for info in info_list])

        episode_rewards = np.zeros((self.n_eval_rollout_threads, self.num_agents))

        self.trainer.prep_rollout()

        for step in range(self.episode_length):
            actions, _, _ = self.trainer.hsd.get_actions_algorithm(
                step,
                obs,
                share_obs,
                available_actions,
                epsilon=0.0
            )
            (
                _obs_single_agent,
                share_obs_next,
                rewards,
                dones,
                infos,
                available_actions_next,
            ) = self.eval_envs.step(actions)

            obs_next, share_obs_next, available_actions_next = self.info_translation(infos)
            episode_rewards += rewards.squeeze(-1)
            obs, share_obs, available_actions = obs_next, share_obs_next, available_actions_next

        for eval_info in infos:
            ep = eval_info["episode"]
            eval_env_infos["eval_sparse_r"].append(ep["ep_sparse_r"])
            eval_env_infos["eval_shaped_r"].append(ep["ep_shaped_r"])

        eval_env_infos["eval_average_episode_rewards"] = np.mean(eval_env_infos["eval_shaped_r"])
        self.log_env(eval_env_infos, total_num_steps)

        eval_env_infos["eval_average_episode_rewards"] = np.mean(episode_rewards)
        self.log_env(eval_env_infos, total_num_steps)

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
                self.trainer.prep_rollout()

                actions, _, _ = self.trainer.hsd.get_actions_algorithm(
                    step,
                    obs,
                    share_obs,
                    available_actions,
                    epsilon=0.0,
                )
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
