import numpy as np
import logging
from typing import Tuple
from okx_stream_hunter.ai.rl_v2.actors.actor_network import ActorNetwork
from okx_stream_hunter.ai.rl_v2.critics.twin_critic import TwinCritic
from okx_stream_hunter.ai.rl_v2.buffers.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class TD3Agent:
    def __init__(self, state_dim: int, action_dim: int, hidden_dims_actor: list = [512, 256, 128],
                 hidden_dims_critic: list = [512, 256, 128], actor_lr: float = 0.0001,
                 critic_lr: float = 0.001, gamma: float = 0.99, tau: float = 0.005,
                 policy_noise: float = 0.2, noise_clip: float = 0.5, policy_delay: int = 2,
                 buffer_capacity: int = 100000, batch_size: int = 256, use_gpu: bool = True):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims_actor, actor_lr, use_gpu)
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden_dims_actor, actor_lr, use_gpu)
        self.actor_target.set_weights(self.actor.get_weights())
        
        self.critic = TwinCritic(state_dim, action_dim, hidden_dims_critic, critic_lr, use_gpu)
        self.critic_target = TwinCritic(state_dim, action_dim, hidden_dims_critic, critic_lr, use_gpu)
        self.critic_target.set_weights(self.critic.get_weights())
        
        self.buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)
        
        self.exploration_noise = 0.1
        self.training_steps = 0
        
        logger.info(f"TD3Agent initialized: state_dim={state_dim}, action_dim={action_dim}, "
                   f"policy_delay={policy_delay}, policy_noise={policy_noise}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        action = self.actor.predict(state, training=False)
        
        if training:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        
        return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        self.buffer.add(state, action, reward, next_state, done)
    
    def train(self) -> dict:
        if not self.buffer.is_ready(self.batch_size):
            return {}
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        noise = np.clip(
            np.random.normal(0, self.policy_noise, size=(self.batch_size, self.action_dim)),
            -self.noise_clip, self.noise_clip
        )
        
        next_actions = self.actor_target.predict(next_states, training=False)
        next_actions = np.clip(next_actions + noise, -1.0, 1.0)
        
        target_q = self.critic_target.predict(next_states, next_actions, use_min=True)
        target_q = rewards + (1 - dones) * self.gamma * target_q
        
        critic_loss = self.critic.train_step(states, actions, target_q)
        
        actor_loss = 0.0
        if self.training_steps % self.policy_delay == 0:
            pred_actions = self.actor.predict(states, training=True)
            q_gradient = self._get_critic_gradient(states, pred_actions)
            actor_loss = self.actor.train_step(states, q_gradient)
            
            self._update_target_networks()
        
        self.training_steps += 1
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'buffer_size': len(self.buffer),
            'training_steps': self.training_steps
        }
    
    def _get_critic_gradient(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if self.critic.critic1.framework == 'tensorflow':
            import tensorflow as tf
            states_tf = tf.constant(states, dtype=tf.float32)
            actions_tf = tf.constant(actions, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(actions_tf)
                q_values = self.critic.critic1.model([states_tf, actions_tf], training=False)
            
            gradients = tape.gradient(q_values, actions_tf)
            return gradients.numpy()
        else:
            import torch
            states_torch = torch.FloatTensor(states).to(self.critic.critic1.device)
            actions_torch = torch.FloatTensor(actions).to(self.critic.critic1.device).requires_grad_(True)
            
            q_values = self.critic.critic1.model(states_torch, actions_torch)
            q_values.backward(torch.ones_like(q_values))
            
            return actions_torch.grad.cpu().numpy()
    
    def _update_target_networks(self):
        actor_weights = self.actor.get_weights()
        actor_target_weights = self.actor_target.get_weights()
        
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
        
        self.actor_target.set_weights(actor_target_weights)
        
        critic_weights = self.critic.get_weights()
        critic_target_weights = self.critic_target.get_weights()
        
        critic_target_weights['critic1'] = [
            self.tau * w + (1 - self.tau) * tw
            for w, tw in zip(critic_weights['critic1'], critic_target_weights['critic1'])
        ]
        critic_target_weights['critic2'] = [
            self.tau * w + (1 - self.tau) * tw
            for w, tw in zip(critic_weights['critic2'], critic_target_weights['critic2'])
        ]
        
        self.critic_target.set_weights(critic_target_weights)
    
    def save(self, filepath_prefix: str):
        self.actor.save(f"{filepath_prefix}_actor.h5")
        self.critic.save(f"{filepath_prefix}_critic1.h5", f"{filepath_prefix}_critic2.h5")
        logger.info(f"TD3 models saved: {filepath_prefix}")
    
    def load(self, filepath_prefix: str):
        self.actor.load(f"{filepath_prefix}_actor.h5")
        self.critic.load(f"{filepath_prefix}_critic1.h5", f"{filepath_prefix}_critic2.h5")
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())
        logger.info(f"TD3 models loaded: {filepath_prefix}")
