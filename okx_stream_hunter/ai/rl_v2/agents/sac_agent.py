import numpy as np
import logging
from typing import Tuple
from okx_stream_hunter.ai.rl_v2.actors.stochastic_actor import StochasticActor
from okx_stream_hunter.ai.rl_v2.critics.twin_critic import TwinCritic
from okx_stream_hunter.ai.rl_v2.buffers.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, hidden_dims_actor: list = [512, 256],
                 hidden_dims_critic: list = [512, 256, 128], actor_lr: float = 0.0003,
                 critic_lr: float = 0.001, alpha_lr: float = 0.0003, gamma: float = 0.99,
                 tau: float = 0.005, alpha: float = 0.2, auto_alpha: bool = True,
                 buffer_capacity: int = 100000, batch_size: int = 256, use_gpu: bool = True):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.auto_alpha = auto_alpha
        
        self.actor = StochasticActor(state_dim, action_dim, hidden_dims_actor, actor_lr, use_gpu=use_gpu)
        
        self.critic = TwinCritic(state_dim, action_dim, hidden_dims_critic, critic_lr, use_gpu)
        self.critic_target = TwinCritic(state_dim, action_dim, hidden_dims_critic, critic_lr, use_gpu)
        self.critic_target.set_weights(self.critic.get_weights())
        
        self.buffer = ReplayBuffer(buffer_capacity, state_dim, action_dim)
        
        if auto_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = np.log(alpha)
            self.alpha_lr = alpha_lr
        else:
            self.alpha = alpha
        
        self.training_steps = 0
        
        logger.info(f"SACAgent initialized: state_dim={state_dim}, action_dim={action_dim}, "
                   f"auto_alpha={auto_alpha}, gamma={gamma}, tau={tau}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        action, _, _ = self.actor.sample_action(state, deterministic=deterministic)
        return action
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        self.buffer.add(state, action, reward, next_state, done)
    
    def train(self) -> dict:
        if not self.buffer.is_ready(self.batch_size):
            return {}
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        if self.actor.framework == 'tensorflow':
            return self._train_tensorflow(states, actions, rewards, next_states, dones)
        else:
            return self._train_pytorch(states, actions, rewards, next_states, dones)
    
    def _train_tensorflow(self, states, actions, rewards, next_states, dones):
        import tensorflow as tf
        
        with tf.GradientTape() as tape:
            next_actions, next_log_probs = self._get_action_log_prob_tf(next_states)
            target_q1, target_q2 = self.critic_target.predict(next_states, next_actions, use_min=False)
            target_q = tf.minimum(target_q1, target_q2)
            
            alpha = tf.exp(self.log_alpha) if self.auto_alpha else self.alpha
            target_q = target_q - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        critic_loss = self.critic.train_step(states, actions, target_q.numpy())
        
        with tf.GradientTape() as tape:
            new_actions, log_probs = self._get_action_log_prob_tf(states)
            q1, q2 = self.critic.predict(states, new_actions, use_min=False)
            q_values = tf.minimum(q1, q2)
            
            alpha = tf.exp(self.log_alpha) if self.auto_alpha else self.alpha
            actor_loss = tf.reduce_mean(alpha * log_probs - q_values)
        
        actor_gradients = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.model.trainable_variables))
        
        alpha_loss = 0.0
        if self.auto_alpha:
            with tf.GradientTape() as tape:
                alpha = tf.exp(self.log_alpha)
                alpha_loss = tf.reduce_mean(-alpha * (log_probs + self.target_entropy))
            
            self.log_alpha -= self.alpha_lr * alpha_loss.numpy()
        
        self._update_target_networks()
        self.training_steps += 1
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': float(actor_loss.numpy()),
            'alpha': float(tf.exp(self.log_alpha).numpy()) if self.auto_alpha else self.alpha,
            'alpha_loss': float(alpha_loss.numpy()) if self.auto_alpha else 0.0,
            'buffer_size': len(self.buffer)
        }
    
    def _train_pytorch(self, states, actions, rewards, next_states, dones):
        import torch
        
        states_torch = torch.FloatTensor(states).to(self.critic.critic1.device)
        actions_torch = torch.FloatTensor(actions).to(self.critic.critic1.device)
        rewards_torch = torch.FloatTensor(rewards).to(self.critic.critic1.device)
        next_states_torch = torch.FloatTensor(next_states).to(self.critic.critic1.device)
        dones_torch = torch.FloatTensor(dones).to(self.critic.critic1.device)
        
        with torch.no_grad():
            next_actions, next_log_probs = self._get_action_log_prob_torch(next_states)
            target_q1, target_q2 = self.critic_target.predict(next_states, next_actions, use_min=False)
            target_q = np.minimum(target_q1, target_q2)
            
            alpha = np.exp(self.log_alpha) if self.auto_alpha else self.alpha
            target_q = target_q - alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        critic_loss = self.critic.train_step(states, actions, target_q)
        
        new_actions, log_probs = self._get_action_log_prob_torch(states)
        q1, q2 = self.critic.predict(states, new_actions, use_min=False)
        q_values = np.minimum(q1, q2)
        
        alpha = np.exp(self.log_alpha) if self.auto_alpha else self.alpha
        actor_loss = np.mean(alpha * log_probs - q_values)
        
        alpha_loss = 0.0
        if self.auto_alpha:
            alpha_loss = -np.mean(alpha * (log_probs + self.target_entropy))
            self.log_alpha -= self.alpha_lr * alpha_loss
        
        self._update_target_networks()
        self.training_steps += 1
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha': np.exp(self.log_alpha) if self.auto_alpha else self.alpha,
            'alpha_loss': alpha_loss if self.auto_alpha else 0.0,
            'buffer_size': len(self.buffer)
        }
    
    def _get_action_log_prob_tf(self, states):
        import tensorflow as tf
        states_tf = tf.constant(states, dtype=tf.float32)
        actions = []
        log_probs = []
        for i in range(states.shape[0]):
            action, log_prob = self.actor.get_action_log_prob(states[i])
            actions.append(action)
            log_probs.append(log_prob)
        return tf.constant(actions, dtype=tf.float32), tf.constant(log_probs, dtype=tf.float32)
    
    def _get_action_log_prob_torch(self, states):
        actions = []
        log_probs = []
        for i in range(states.shape[0]):
            action, log_prob = self.actor.get_action_log_prob(states[i])
            actions.append(action)
            log_probs.append(log_prob)
        return np.array(actions), np.array(log_probs)
    
    def _update_target_networks(self):
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
        logger.info(f"SAC models saved: {filepath_prefix}")
    
    def load(self, filepath_prefix: str):
        self.actor.load(f"{filepath_prefix}_actor.h5")
        self.critic.load(f"{filepath_prefix}_critic1.h5", f"{filepath_prefix}_critic2.h5")
        self.critic_target.set_weights(self.critic.get_weights())
        logger.info(f"SAC models loaded: {filepath_prefix}")
