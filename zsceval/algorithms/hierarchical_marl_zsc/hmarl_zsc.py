import torch
from loguru import logger

from zsceval.algorithms.hierarchical_marl_zsc.hmarl_zsc_actor_critic import R_Actor, R_Critic
from zsceval.utils.util import update_linear_schedule
from zsceval.algorithms.hierarchical_marl_zsc.hmarl_trainer import HMARLTrainer

class ExDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class Ego_HMARLPolicy:
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.data_parallel = getattr(args, "data_parallel", False)
        
        """ 
        Manager is only trainable part of Ego Agent. It sees share_obs_space (trajectory of other partners, env, information from warmup) and outputs skill dimension action periodically.
        
        New skill is generated periodically, and Ego agents deterministically rollouts low level action only based on the given skill from Manager.

        Actor: 
        - At start of each period (internal counter%skill_period=0), gives skill as action, based on share_obs_space and information from warmup & last period and stores it as internal variable. 
        - During the period (internal counter%skill_period!=0), it just returns the deterministic action based on skill & pretrained policy. It also collects trajectory of partners during the period for next period's skill generation.
        - Internal algorithm:
            1. If internal counter%skill_period=0, receives trajectory of partner from last period, skill of current ego agent from last period. Creates new skill and store it internally.
                Info from last period used for new skill generation (Selective):
                    - Skill_estimate: Predicts current skill of partners using skill = pretrained_decoder(trajectory from last period)
                    - Skill_ego: Actual skill of current ego agent from last period.
                    - RNN_general_encoding: Encode trajectory of each agent using RNN
                    - Partner_Behavior: Encode behavior of partner using (skill of partner|skills of others(ego + other partners))
                Prediction of new skill:
                    - Use neural network that takes above info
                Interal State Update:
                    - Update internal counter
                    - Update internal skill with new skill
                    - Reset trajectory buffer for next period
                Store buffer for training Actor:
                    - Store (encodings of Info, new skill) into replay buffer for training Manager actor.

            2. If internal counter%skill_period!=0, receives current observation, based on stored internal skill, outputs low level action deterministically based on pretrained low level policy.
                (Note: Low level policy might act in detail based on its input Q-function)
                Deterministic low level action generation:
                    - Use pretrained low level policy that takes (current observation, internal skill)
                Internal State Update:
                    - Update internal counter
                    - Store trajectory of partners during the period for next period's skill generation.

            3. If step%(training_step*skill_period)=0, train the Manager actor & critic using buffer. Reset the buffer after training.

        Critic:
        - Gives value function for skill dimension action based on share_obs_space (trajectory of other partners, env, information from warmup)
        - Critic Training Period is same as with Actor Training Period.
        """

        # 문기가 API Spec 적어서 이현님 R_Actor, R_Critic 완성 (hmarl_zsc_actor_critic.py)
        self.actor = R_Actor(args, self.obs_space, self.num_skills, self.device) # FIXME: change R_Actor with our own Manager Actor: Should include 
        self.critic = R_Critic(args, self.share_obs_space, self.device) # FIXME: change R_Critic with our own Manager Critic
        self.pretrained_hsd = HMARLTrainer(args, obs_space, share_obs_space, act_space, device) # FIXME: fix parameter inputs for HMARLTrainer

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def load_HMARLTrainer(self, hmarl_path):
        self.pretrained_hsd.load_checkpoint(hmarl_path)

    def to_parallel(self):
        if self.data_parallel:
            logger.warning(
                f"Use Data Parallel for Forwarding in devices {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
            )
            for name, children in self.actor.named_children():
                setattr(self.actor, name, ExDataParallel(children)) # just wrapping submodule of module into multi-gpu just for here
            for name, children in self.critic.named_children():
                setattr(self.critic, name, ExDataParallel(children)) # additional syntax not required

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(
        self,
        share_obs,
        obs,
        hidden_states_actor, # internal hidden state of actor RNN
        hidden_states_critic, # internal hidden state of critic RNN
        masks,
        available_actions=None,
        deterministic=False,
        task_id=None,
        **kwargs,
    ):
        actions, action_log_probs, hidden_states_actor = self.actor(
            obs, hidden_states_actor, masks, available_actions, deterministic
        )
        values, hidden_states_critic = self.critic(share_obs, hidden_states_critic, masks, task_id=task_id)
        return values, actions, action_log_probs, hidden_states_actor, hidden_states_critic

    def get_values(self, share_obs, hidden_states_critic, masks, task_id=None):
        values, _ = self.critic(share_obs, hidden_states_critic, masks, task_id=task_id)
        return values

    def evaluate_actions(
        self,
        share_obs,
        obs,
        hidden_states_actor,
        hidden_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        task_id=None,
    ):
        (
            action_log_probs,
            dist_entropy,
            policy_values,
        ) = self.actor.evaluate_actions(obs, hidden_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(share_obs, hidden_states_critic, masks, task_id=task_id)
        return values, action_log_probs, dist_entropy, policy_values

    def evaluate_transitions(
        self,
        share_obs,
        obs,
        hidden_states_actor,
        hidden_states_critic,
        action,
        masks,
        available_actions=None,
        active_masks=None,
        task_id=None,
    ):
        (
            action_log_probs,
            dist_entropy,
            policy_values,
            hidden_states_actor,
        ) = self.actor.evaluate_transitions(obs, hidden_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(share_obs, hidden_states_critic, masks, task_id=task_id)
        return values, action_log_probs, dist_entropy, policy_values, hidden_states_actor

    def act(
        self,
        obs,
        hidden_states_actor,
        masks,
        available_actions=None,
        deterministic=False,
        **kwargs,
    ):
        actions, _, hidden_states_actor = self.actor(obs, hidden_states_actor, masks, available_actions, deterministic)
        return actions, hidden_states_actor

    def get_probs(self, obs, hidden_states_actor, masks, available_actions=None):
        action_probs, hidden_states_actor = self.actor.get_probs(
            obs, hidden_states_actor, masks, available_actions=available_actions
        )
        return action_probs, hidden_states_actor

    def get_action_log_probs(
        self,
        obs,
        hidden_states_actor,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        action_log_probs, _, _, hidden_states_actor = self.actor.get_action_log_probs(
            obs, hidden_states_actor, action, masks, available_actions, active_masks
        )
        return action_log_probs, hidden_states_actor

    def load_checkpoint(self, ckpt_path):
        if "actor" in ckpt_path:
            self.actor.load_state_dict(torch.load(ckpt_path["actor"], map_location=self.device))
        if "critic" in ckpt_path:
            self.critic.load_state_dict(torch.load(ckpt_path["critic"], map_location=self.device))

    def to(self, device):
        self.actor.to(device)
        self.critic.to(device)

    def prep_rollout(self):
        self.actor.eval()
        self.critic.eval()
