import numpy as np
import logging
from typing import Tuple
from okx_stream_hunter.ai.rl_v2.actors.actor_network import ActorNetwork
from okx_stream_hunter.ai.rl_v2.critics.critic_network import CriticNetwork
from okx_stream_hunter.ai.rl_v2.buffers.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, hidden_dims_actor: list = [512, 256, 128],
                 hidden_dims_critic: list = [512, 256, 128], actor_lr: float = 0.0001,
                 critic_lr: float = 0.001, gamma: float = 0.99, tau: float = 0.005,
                 buffer_capacity: int = 100000, batch_size: int = 256, use_gpu: bool = True):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims_actor, actor_lr, use_gpu)
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden_dims_actor, actor_lr, use_gpu)
        self.actor_target.set_weights(self.actor.get_weights())
        
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dims_critic, critic_lr, use_gpu)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_dims_critic, critic_lr, use_gpu)
        self.critic_target.set_weights(self.critic.get_weights())
        
        self.buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)
        
        self.noise_scale = 0.2
        self.noise_decay = 0.9995
        self.min_noise = 0.01
        
        self.training_steps = 0
        
        logger.info(f"DDPGAgent initialized: state_dim={state_dim}, action_dim={action_dim}, "
                   f"gamma={gamma}, tau={tau}, batch_size={batch_size}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        action = self.actor.predict(state, training=False)
        
        if training:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
            self.noise_scale = max(self.noise_scale * self.noise_decay, self.min_noise)
        
        return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        self.buffer.add(state, action, reward, next_state, done)
    
    def train(self) -> dict:
        if not self.buffer.is_ready(self.batch_size):
            return {}
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        next_actions = self.actor_target.predict(next_states, training=False)
        target_q = self.critic_target.predict(next_states, next_actions)
        target_q = rewards + (1 - dones) * self.gamma * target_q
        
        critic_loss = self.critic.train_step(states, actions, target_q)
        
        pred_actions = self.actor.predict(states, training=True)
        q_gradient = self._get_critic_gradient(states, pred_actions)
        actor_loss = self.actor.train_step(states, q_gradient)
        
        self._update_target_networks()
        
        self.training_steps += 1
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'noise_scale': self.noise_scale,
            'buffer_size': len(self.buffer)
        }
    
    def _get_critic_gradient(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if self.critic.framework == 'tensorflow':
            import tensorflow as tf
            states_tf = tf.constant(states, dtype=tf.float32)
            actions_tf = tf.constant(actions, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(actions_tf)
                q_values = self.critic.model([states_tf, actions_tf], training=False)
            
            gradients = tape.gradient(q_values, actions_tf)
            return gradients.numpy()
        else:
            import torch
            states_torch = torch.FloatTensor(states).to(self.critic.device)
            actions_torch = torch.FloatTensor(actions).to(self.critic.device).requires_grad_(True)
            
            q_values = self.critic.model(states_torch, actions_torch)
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
        
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
        
        self.critic_target.set_weights(critic_target_weights)
    
    def save(self, filepath_prefix: str):
        self.actor.save(f"{filepath_prefix}_actor.h5")
        self.critic.save(f"{filepath_prefix}_critic.h5")
        logger.info(f"DDPG models saved: {filepath_prefix}")
    
    def load(self, filepath_prefix: str):
        self.actor.load(f"{filepath_prefix}_actor.h5")
        self.critic.load(f"{filepath_prefix}_critic.h5")
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())
        logger.info(f"DDPG models loaded: {filepath_prefix}")
