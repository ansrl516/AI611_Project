#!/usr/bin/env python
"""
Post-hoc Evaluation Script for Overcooked

Loads checkpoints and evaluates agents in a specified layout, saving GIF animations.

Usage:
    python -m zsceval.scripts.overcooked.eval.eval_posthoc \
        --layout_name counter_circuit_o_1order \
        --checkpoint_path /home/bml/neurocontroller/results/Overcooked/random0_medium/rmappo/sp/seed5/models/actor_agent0_periodic_10000000.pt \
        --num_agents 3 \
        --num_episodes 1 \
        --use_render
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# Add project root to path if needed
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from zsceval.config import get_config
from zsceval.overcooked_config import get_overcooked_args
from zsceval.envs.env_wrappers import ShareDummyVecEnv
from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked as Overcooked_new
from zsceval.runner.separated.base_runner import make_trainer_policy_cls, _t2n
from zsceval.utils.train_util import setup_seed


def make_eval_env(all_args, run_dir):
    """Create evaluation environment wrapped in VecEnv."""
    def get_env_fn(rank):
        def init_env():
            # Set featurize_type for N agents
            featurize_type = tuple(["ppo"] * all_args.num_agents)
            env = Overcooked_new(all_args, run_dir, rank=rank, evaluation=True, featurize_type=featurize_type)
            env.seed(all_args.seed * 50000 + rank * 10000)
            # Enable rendering if requested
            if all_args.use_render:
                env.use_render = True
                # Make sure gifs directory exists
                gif_dir = Path(run_dir) / "gifs"
                gif_dir.mkdir(parents=True, exist_ok=True)
            return env
        return init_env
    
    return ShareDummyVecEnv([get_env_fn(0)])


def load_checkpoint(policy, checkpoint_path, device, agent_id):
    """Load actor and critic checkpoints into policy."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load actor
    actor_state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy.actor.load_state_dict(actor_state_dict)
    policy.actor.eval()
    logger.info(f"  ✓ Agent {agent_id}: Loaded actor from {checkpoint_path.name}")
    
    # Load critic if available
    critic_path = checkpoint_path.parent / checkpoint_path.name.replace("actor", "critic")
    if critic_path.exists():
        critic_state_dict = torch.load(critic_path, map_location=device, weights_only=False)
        policy.critic.load_state_dict(critic_state_dict)
        policy.critic.eval()
        logger.info(f"  ✓ Agent {agent_id}: Loaded critic from {critic_path.name}")


def run_episode(eval_envs, trainers, all_args, episode_id):
    """Run a single episode and return trajectory info."""
    num_agents = all_args.num_agents
    n_rollout_threads = all_args.n_eval_rollout_threads
    
    # Reset environment
    eval_obs_batch, eval_info_list = eval_envs.reset()
    eval_obs = np.array([info['all_agent_obs'] for info in eval_info_list])
    # Get available_actions from info, with fallback
    eval_available_actions_list = [info.get('available_actions', None) for info in eval_info_list]
    if eval_available_actions_list[0] is not None:
        eval_available_actions = np.array(eval_available_actions_list)
    else:
        # Fallback: create default available actions (all actions available)
        action_dim = eval_envs.action_space[0].n if hasattr(eval_envs.action_space[0], 'n') else 6
        eval_available_actions = np.ones((n_rollout_threads, num_agents, action_dim), dtype=np.int32)
    
    # Initialize RNN states
    eval_rnn_states = np.zeros(
        (n_rollout_threads, num_agents, all_args.recurrent_N, all_args.hidden_size),
        dtype=np.float32
    )
    eval_masks = np.ones((n_rollout_threads, num_agents, 1), dtype=np.float32)
    
    episode_rewards = []
    episode_length = 0
    
    for step in range(all_args.episode_length):
        eval_actions = []
        
        # Get actions from all agents
        for agent_id in range(num_agents):
            trainer = trainers[agent_id]
            trainer.prep_rollout()
            
            # Use policy.act() method
            eval_action, eval_rnn_state = trainer.policy.act(
                eval_obs[:, agent_id],  # Shape: (n_rollout_threads, H, W, C)
                eval_rnn_states[:, agent_id],  # Shape: (n_rollout_threads, recurrent_N, hidden_size)
                eval_masks[:, agent_id],  # Shape: (n_rollout_threads, 1)
                eval_available_actions[:, agent_id],  # Shape: (n_rollout_threads, action_dim)
                deterministic=True
            )
            
            eval_action = _t2n(eval_action)
            eval_actions.append(eval_action)
            eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
        
        # Stack actions: (num_agents, n_rollout_threads, action_dim) -> (n_rollout_threads, num_agents, action_dim)
        eval_actions = np.stack(eval_actions).transpose(1, 0, 2)
        
        # Environment step
        (
            _eval_obs_batch_single_agent,
            _,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = eval_envs.step(eval_actions)
        
        # Extract observations
        eval_obs = np.array([info['all_agent_obs'] for info in eval_infos])
        # Update available_actions from info
        available_actions_list = [info.get('available_actions', None) for info in eval_infos]
        if available_actions_list[0] is not None:
            eval_available_actions = np.array(available_actions_list)
        
        # Store rewards
        episode_rewards.append(eval_rewards[0])
        episode_length += 1
        
        # Update RNN states and masks for done episodes
        eval_rnn_states[eval_dones == True] = np.zeros(
            ((eval_dones == True).sum(), all_args.recurrent_N, all_args.hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones((n_rollout_threads, num_agents, 1), dtype=np.float32)
        eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
        
        # Check if episode is done
        if np.all(eval_dones):
            break
    
    # Render GIF if requested
    if all_args.use_render:
        try:
            # Get the underlying environment from VecEnv
            env = eval_envs.envs[0]  # ShareDummyVecEnv uses self.envs
            env.render()
            logger.info(f"✓ GIF saved for episode {episode_id}")
        except Exception as e:
            logger.warning(f"Failed to render GIF: {e}")
    
    total_reward = np.sum(episode_rewards) if episode_rewards else 0.0
    return {
        'episode_id': episode_id,
        'total_reward': total_reward,
        'episode_length': episode_length,
        'rewards_by_agent': np.sum(episode_rewards, axis=0) if episode_rewards else np.zeros(num_agents)
    }


def main():
    import sys
    import argparse
    
    # First, parse only the custom arguments we need from command line
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint file")
    cmd_parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run")
    cmd_parser.add_argument("--layout_name", type=str, default="counter_circuit_o_1order", help="Layout name")
    cmd_parser.add_argument("--num_agents", type=int, default=3, help="Number of agents")
    cmd_parser.add_argument("--seed", type=int, default=5, help="Random seed")
    cmd_parser.add_argument("--use_render", action="store_true", default=True, help="Enable rendering")
    cmd_parser.add_argument("--cuda", action="store_true", default=False, help="Use CUDA")
    
    cmd_args, _ = cmd_parser.parse_known_args()
    
    # Get values from command line or use defaults
    layout_name = cmd_args.layout_name
    num_agents = cmd_args.num_agents
    seed = cmd_args.seed
    checkpoint_path = cmd_args.checkpoint_path
    num_episodes = cmd_args.num_episodes
    use_render = cmd_args.use_render
    use_cuda = cmd_args.cuda
    
    # Now create the full parser for all_args
    parser = get_config()
    parser = get_overcooked_args(parser)
    
    # 필수 인자 설정 (train_sp.sh의 설정을 그대로 사용)
    args_list = [
        "--algorithm_name", "rmappo",
        "--env_name", "Overcooked",
        "--layout_name", layout_name,
        "--experiment_name", "post_hoc_eval",
        "--seed", str(seed),
        "--overcooked_version", "new",
        "--num_agents", str(num_agents),
        "--n_eval_rollout_threads", "1",
        "--episode_length", "400",
        "--use_recurrent_policy",  # Boolean 플래그는 값 없이
        # --use_centralized_V는 action="store_false", default=True이므로 플래그가 없으면 True
        # train_sp.sh에는 없으므로 기본값 True 사용 (플래그 추가하지 않음)
        "--cnn_layers_params", "32,3,1,1 64,3,1,1 32,3,1,1",  # train_sp.sh에서 사용된 설정
        "--random_index",  # train_sp.sh에서 사용됨
        "--recurrent_N", "1",  # RNN layers
        # hidden_size는 명시되지 않았으므로 기본값 64 사용
        # use_agent_policy_id는 train_sp.sh에 없으므로 기본값 False 사용
    ]
    
    # Parse args_list to get base configuration
    all_args = parser.parse_args(args_list)
    
    # Override with command line values
    all_args.checkpoint_path = checkpoint_path
    all_args.num_episodes = num_episodes
    
    # 추가 설정 (parse_args 후에 직접 설정)
    all_args.use_eval = True
    all_args.eval_stochastic = False  # Deterministic evaluation
    all_args.use_wandb = False
    all_args.use_render = use_render  # GIF 렌더링을 위해 필요
    all_args.n_render_rollout_threads = 1  # 렌더링 스레드 수
    all_args.cuda = use_cuda  # CPU 사용 (필요시 True로 변경)
    all_args.use_phi = False  # use_phi 사용 안 함
    all_args.use_hsp = False  # use_hsp 사용 안 함
    all_args.old_dynamics = False  # old_dynamics 사용 안 함 (new version 사용)
    all_args.use_agent_policy_id = False  # train_sp.sh에 없으므로 False로 명시적 설정
    # use_centralized_V는 train_sp.sh에 없으므로 기본값 True 사용
    # 하지만 parse_args에서 action="store_false"이므로 플래그가 없으면 True가 됨
    # 명시적으로 True로 설정하여 확실하게 함
    all_args.use_centralized_V = True
    all_args.use_available_actions = True
    
    # agent_policy_names 처리: rMAPPO 정책을 직접 로드하므로 scripted agent 불필요
    # parse_args()에서 default=None으로 설정되지만, 환경 코드가 len(None)을 호출하려고 해서 에러 발생
    # 따라서 None이 아닌 빈 리스트로 설정하여 환경 코드의 len() 호출을 안전하게 만듦
    if not hasattr(all_args, 'agent_policy_names') or all_args.agent_policy_names is None:
        all_args.agent_policy_names = [None] * all_args.num_agents  # num_agents 길이만큼 None으로 채움 (scripted agent 사용 안 함)
    
    # Additional required attributes for environment initialization
    all_args.initial_reward_shaping_factor = getattr(all_args, 'initial_reward_shaping_factor', 1.0)
    all_args.reward_shaping_factor = getattr(all_args, 'reward_shaping_factor', 1.0)
    all_args.reward_shaping_horizon = getattr(all_args, 'reward_shaping_horizon', 2.5e6)
    all_args.use_random_player_pos = getattr(all_args, 'use_random_player_pos', False)
    all_args.use_random_terrain_state = getattr(all_args, 'use_random_terrain_state', False)
    all_args.num_initial_state = getattr(all_args, 'num_initial_state', 1)
    all_args.replay_return_threshold = getattr(all_args, 'replay_return_threshold', 0)
    
    # Setup
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        logger.info("Using GPU")
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
        logger.info("Using CPU")
    
    # Setup seed
    setup_seed(all_args.seed)
    
    # Create run directory
    run_dir = Path(all_args.run_dir) if hasattr(all_args, 'run_dir') and all_args.run_dir else Path("./eval_results")
    run_dir = run_dir / all_args.layout_name / f"episodes_{all_args.num_episodes}"
    run_dir.mkdir(parents=True, exist_ok=True)
    all_args.run_dir = str(run_dir)
    logger.info(f"Results will be saved to: {run_dir}")
    
    # Create evaluation environment
    logger.info(f"Creating environment: {all_args.layout_name} with {all_args.num_agents} agents")
    eval_envs = make_eval_env(all_args, run_dir)
    
    # Get trainer and policy classes
    TrainAlgo, Policy = make_trainer_policy_cls(all_args.algorithm_name)
    
    # Create policies and trainers
    logger.info("Initializing policies and loading checkpoints...")
    policies = []
    trainers = []
    
    checkpoint_paths = [
        all_args.checkpoint_path,  # agent0
        all_args.checkpoint_path,  # agent1 (same as agent0)
        all_args.checkpoint_path,  # agent2 (same as agent0)
    ]
    
    # Adjust checkpoint paths for agent1 and agent2 if different files exist
    checkpoint_dir = Path(all_args.checkpoint_path).parent
    for agent_id in range(1, all_args.num_agents):
        agent_checkpoint = checkpoint_dir / f"actor_agent{agent_id}_periodic_10000000.pt"
        if agent_checkpoint.exists():
            checkpoint_paths[agent_id] = str(agent_checkpoint)
            logger.info(f"  Using agent{agent_id} specific checkpoint: {agent_checkpoint.name}")
        else:
            logger.info(f"  Agent {agent_id} using agent0 checkpoint")
    
    for agent_id in range(all_args.num_agents):
        # Create policy
        share_observation_space = (
            eval_envs.share_observation_space[agent_id]
            if all_args.use_centralized_V
            else eval_envs.observation_space[agent_id]
        )
        
        policy = Policy(
            all_args,
            eval_envs.observation_space[agent_id],
            share_observation_space,
            eval_envs.action_space[agent_id],
            device=device
        )
        policies.append(policy)
        
        # Load checkpoint
        load_checkpoint(policy, checkpoint_paths[agent_id], device, agent_id)
        
        # Create trainer
        trainer = TrainAlgo(all_args, policy, device=device)
        trainers.append(trainer)
    
    logger.info(f"✓ Initialized {len(policies)} policies and trainers")
    
    # Run episodes
    logger.info(f"\nRunning {all_args.num_episodes} episode(s)...")
    results = []
    
    for episode_id in range(all_args.num_episodes):
        logger.info(f"\nEpisode {episode_id + 1}/{all_args.num_episodes}")
        result = run_episode(eval_envs, trainers, all_args, episode_id)
        results.append(result)
        
        logger.info(f"  Episode {episode_id + 1} completed:")
        logger.info(f"    Total reward: {result['total_reward']:.2f}")
        logger.info(f"    Episode length: {result['episode_length']}")
        logger.info(f"    Rewards by agent: {result['rewards_by_agent']}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Evaluation Summary")
    logger.info("="*60)
    avg_reward = np.mean([r['total_reward'] for r in results])
    avg_length = np.mean([r['episode_length'] for r in results])
    logger.info(f"Average total reward: {avg_reward:.2f}")
    logger.info(f"Average episode length: {avg_length:.1f}")
    logger.info(f"GIFs saved to: {run_dir / 'gifs'}")
    
    # Cleanup
    eval_envs.close()
    
    return 0


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    sys.exit(main())

