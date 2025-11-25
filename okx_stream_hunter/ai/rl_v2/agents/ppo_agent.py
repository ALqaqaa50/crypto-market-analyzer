import numpy as np
import logging
from typing import Dict, List
from okx_stream_hunter.ai.rl_v2.buffers.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [512, 256],
                 actor_lr: float = 0.0003, critic_lr: float = 0.001, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_epsilon: float = 0.2, epochs: int = 10,
                 batch_size: int = 64, use_gpu: bool = True):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        
        if HAS_TF:
            self.framework = 'tensorflow'
            self._build_tf_models(hidden_dims, actor_lr, critic_lr, use_gpu)
        elif HAS_TORCH:
            self.framework = 'pytorch'
            self._build_torch_models(hidden_dims, actor_lr, critic_lr, use_gpu)
        else:
            raise ImportError("Neither TensorFlow nor PyTorch available")
        
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        logger.info(f"PPOAgent initialized: framework={self.framework}, clip_epsilon={clip_epsilon}")
    
    def _build_tf_models(self, hidden_dims, actor_lr, critic_lr, use_gpu):
        inputs = keras.Input(shape=(self.state_dim,))
        x = inputs
        for dim in hidden_dims:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.LayerNormalization()(x)
        
        mean = layers.Dense(self.action_dim)(x)
        log_std = layers.Dense(self.action_dim)(x)
        
        self.actor = keras.Model(inputs=inputs, outputs=[mean, log_std])
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=actor_lr)
        
        inputs_critic = keras.Input(shape=(self.state_dim,))
        x = inputs_critic
        for dim in hidden_dims:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.LayerNormalization()(x)
        value = layers.Dense(1)(x)
        
        self.critic = keras.Model(inputs=inputs_critic, outputs=value)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=critic_lr)
    
    def _build_torch_models(self, hidden_dims, actor_lr, critic_lr, use_gpu):
        class TorchPPOActor(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dims):
                super(TorchPPOActor, self).__init__()
                layers_list = []
                prev_dim = state_dim
                for dim in hidden_dims:
                    layers_list.append(nn.Linear(prev_dim, dim))
                    layers_list.append(nn.ReLU())
                    layers_list.append(nn.LayerNorm(dim))
                    prev_dim = dim
                self.shared = nn.Sequential(*layers_list)
                self.mean_layer = nn.Linear(prev_dim, action_dim)
                self.log_std_layer = nn.Linear(prev_dim, action_dim)
            
            def forward(self, state):
                x = self.shared(state)
                return self.mean_layer(x), self.log_std_layer(x)
        
        class TorchPPOCritic(nn.Module):
            def __init__(self, state_dim, hidden_dims):
                super(TorchPPOCritic, self).__init__()
                layers_list = []
                prev_dim = state_dim
                for dim in hidden_dims:
                    layers_list.append(nn.Linear(prev_dim, dim))
                    layers_list.append(nn.ReLU())
                    layers_list.append(nn.LayerNorm(dim))
                    prev_dim = dim
                layers_list.append(nn.Linear(prev_dim, 1))
                self.network = nn.Sequential(*layers_list)
            
            def forward(self, state):
                return self.network(state)
        
        self.actor = TorchPPOActor(self.state_dim, self.action_dim, hidden_dims)
        self.critic = TorchPPOCritic(self.state_dim, hidden_dims)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        if use_gpu and torch.cuda.is_available():
            self.actor = self.actor.cuda()
            self.critic = self.critic.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    def select_action(self, state: np.ndarray):
        if self.framework == 'tensorflow':
            import tensorflow_probability as tfp
            if len(state.shape) == 1:
                state = np.expand_dims(state, axis=0)
            mean, log_std = self.actor(state)
            std = tf.exp(log_std)
            dist = tfp.distributions.Normal(mean, std)
            action = dist.sample()
            action = tf.tanh(action)
            log_prob = dist.log_prob(action)
            log_prob = tf.reduce_sum(log_prob, axis=1)
            value = self.critic(state)
            return action.numpy()[0], log_prob.numpy()[0], value.numpy()[0][0]
        else:
            from torch.distributions import Normal
            state_tensor = torch.FloatTensor(state).to(self.device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            self.actor.eval()
            self.critic.eval()
            with torch.no_grad():
                mean, log_std = self.actor(state_tensor)
                std = torch.exp(log_std)
                dist = Normal(mean, std)
                action = dist.sample()
                action = torch.tanh(action)
                log_prob = dist.log_prob(action).sum(dim=1)
                value = self.critic(state_tensor)
            return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0][0]
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dones'].append(done)
    
    def train(self) -> dict:
        if len(self.buffer['states']) == 0:
            return {}
        
        states = np.array(self.buffer['states'])
        actions = np.array(self.buffer['actions'])
        rewards = np.array(self.buffer['rewards'])
        values = np.array(self.buffer['values'])
        old_log_probs = np.array(self.buffer['log_probs'])
        dones = np.array(self.buffer['dones'])
        
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        actor_losses = []
        critic_losses = []
        
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                if self.framework == 'tensorflow':
                    actor_loss, critic_loss = self._train_step_tf(
                        batch_states, batch_actions, batch_old_log_probs,
                        batch_advantages, batch_returns
                    )
                else:
                    actor_loss, critic_loss = self._train_step_torch(
                        batch_states, batch_actions, batch_old_log_probs,
                        batch_advantages, batch_returns
                    )
                
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
        
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'epochs': self.epochs
        }
    
    def _compute_gae(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
        
        return advantages
    
    def _train_step_tf(self, states, actions, old_log_probs, advantages, returns):
        import tensorflow_probability as tfp
        
        with tf.GradientTape() as actor_tape:
            mean, log_std = self.actor(states, training=True)
            std = tf.exp(log_std)
            dist = tfp.distributions.Normal(mean, std)
            new_log_probs = tf.reduce_sum(dist.log_prob(actions), axis=1)
            
            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            
            actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
        
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        with tf.GradientTape() as critic_tape:
            pred_values = self.critic(states, training=True)
            critic_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(pred_values)))
        
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        return float(actor_loss.numpy()), float(critic_loss.numpy())
    
    def _train_step_torch(self, states, actions, old_log_probs, advantages, returns):
        from torch.distributions import Normal
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device).unsqueeze(1)
        
        self.actor.train()
        self.actor_optimizer.zero_grad()
        
        mean, log_std = self.actor(states_tensor)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        new_log_probs = dist.log_prob(actions_tensor).sum(dim=1)
        
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        actor_loss = -torch.mean(torch.min(ratio * advantages_tensor, clipped_ratio * advantages_tensor))
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic.train()
        self.critic_optimizer.zero_grad()
        
        pred_values = self.critic(states_tensor)
        critic_loss = F.mse_loss(pred_values, returns_tensor)
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return float(actor_loss.item()), float(critic_loss.item())
    
    def save(self, filepath_prefix: str):
        if self.framework == 'tensorflow':
            self.actor.save_weights(f"{filepath_prefix}_actor.h5")
            self.critic.save_weights(f"{filepath_prefix}_critic.h5")
        else:
            torch.save(self.actor.state_dict(), f"{filepath_prefix}_actor.pth")
            torch.save(self.critic.state_dict(), f"{filepath_prefix}_critic.pth")
        logger.info(f"PPO models saved: {filepath_prefix}")
    
    def load(self, filepath_prefix: str):
        if self.framework == 'tensorflow':
            self.actor.load_weights(f"{filepath_prefix}_actor.h5")
            self.critic.load_weights(f"{filepath_prefix}_critic.h5")
        else:
            self.actor.load_state_dict(torch.load(f"{filepath_prefix}_actor.pth", map_location=self.device))
            self.critic.load_state_dict(torch.load(f"{filepath_prefix}_critic.pth", map_location=self.device))
        logger.info(f"PPO models loaded: {filepath_prefix}")
