#!/bin/bash
env="Overcooked_new"

layout="cramped_room"
cuda_device=${2:-0}  # Second argument for CUDA device, default to 0
# Always use NEW version for all layouts
version="new"

entropy_coefs="0.2 0.05 0.01"
entropy_coef_horizons="0 5e6 1e7"
if [[ "${layout}" == "small_corridor" ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 8e6 1e7"
fi

reward_shaping_horizon="1e8"
num_env_steps="1e7"

num_agents=3
algo="hmarl"
exp="sp"
seed_begin=11
seed_max=15
ulimit -n 65536

# Heterogeneous Shaped Reward weights (30 components for new version)
# 세 에이전트 모두 같은 weights 사용
# w0="0,0,0,-1,0.1,0,0,0,0.1,0,-1,0.5,1.0,0.2,0.1,0.2,-2,-1,0.2,0,0.5,50,0,20,-2,-0.01,-0.01,-0.1,-0.1,50"
# w1="0.1,0,0.2,0,0,0.1,0,0,0,0.2,0,0,0,0,0,0,0,-1,0,0,0,50,0,0,0,-0.01,-0.01,-0.1,-0.1,50"
# w2="0,0,0,-1,0.1,0,0,0,0.1,0,-1,0.5,1.0,0.2,0.1,0.2,-2,-1,0.2,0,0.5,50,0,20,-2,-0.01,-0.01,-0.1,-0.1,50"
w0="0,0,0,0,0,0.1,0.1,0,0,0.1,0,3,0,10,-2,3,2,2,-2,-2,5,5,0,20,-5,0,7,20,-5,-0.01,-0.01,-0.1,-0.1,30"
w1="0,0,0,0,0,0.1,0.1,0,0,0.1,0,3,0,10,-2,3,2,2,-2,-2,5,5,0,20,-5,0,7,20,-5,-0.01,-0.01,-0.1,-0.1,30"

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, seed from ${seed_begin} to ${seed_max}"
echo "Using CUDA device: ${cuda_device}"
for seed in $(seq ${seed_begin} ${seed_max});
do
    echo "seed is ${seed}:"
    python ../train/train_hmarl.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --layout_name ${layout} --num_agents ${num_agents} \
    --agent_policy_names ppo ppo ppo \ # number of agents에 맞게 policy names 지정
    --seed ${seed} --n_training_threads 1 --n_rollout_threads 50 --dummy_batch_size 2 --num_mini_batch 1 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    --overcooked_version ${version} \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --use_hsp --w0 ${w0} --w1 ${w1} --share_policy --random_index \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy \
    --use_proper_time_limits \
    --save_interval 25 --log_interval 10 --use_eval --eval_interval 20 --n_eval_rollout_threads 10 \
    --use_render --save_gifs --n_render_rollout_threads 1 --render_episodes 1 \
    --cuda_device ${cuda_device} 
    # --wandb_name "kyungyoon"
done
