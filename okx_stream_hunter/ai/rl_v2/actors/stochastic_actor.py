import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow_probability as tfp
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class StochasticActor:
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [512, 256],
                 learning_rate: float = 0.0003, log_std_min: float = -20, log_std_max: float = 2,
                 use_gpu: bool = True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.use_gpu = use_gpu
        
        if HAS_TF:
            self.framework = 'tensorflow'
            self._build_tf_model()
        elif HAS_TORCH:
            self.framework = 'pytorch'
            self._build_torch_model()
        else:
            raise ImportError("Neither TensorFlow nor PyTorch available")
        
        logger.info(f"StochasticActor initialized: framework={self.framework}")
    
    def _build_tf_model(self):
        inputs = keras.Input(shape=(self.state_dim,))
        x = inputs
        
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.LayerNormalization()(x)
        
        mean = layers.Dense(self.action_dim)(x)
        log_std = layers.Dense(self.action_dim)(x)
        
        self.model = keras.Model(inputs=inputs, outputs=[mean, log_std])
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
    
    def _build_torch_model(self):
        class TorchStochasticActor(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dims):
                super(TorchStochasticActor, self).__init__()
                
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
                mean = self.mean_layer(x)
                log_std = self.log_std_layer(x)
                return mean, log_std
        
        self.model = TorchStochasticActor(self.state_dim, self.action_dim, self.hidden_dims)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        if self.use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    def sample_action(self, state: np.ndarray, deterministic: bool = False):
        if self.framework == 'tensorflow':
            if len(state.shape) == 1:
                state = np.expand_dims(state, axis=0)
            
            mean, log_std = self.model(state)
            log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
            std = tf.exp(log_std)
            
            if deterministic:
                action = mean
            else:
                normal = tfp.distributions.Normal(mean, std)
                action = normal.sample()
            
            action = tf.tanh(action)
            return action.numpy()[0], mean.numpy()[0], log_std.numpy()[0]
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            self.model.eval()
            with torch.no_grad():
                mean, log_std = self.model(state_tensor)
                log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
                std = torch.exp(log_std)
                
                if deterministic:
                    action = mean
                else:
                    normal = Normal(mean, std)
                    action = normal.sample()
                
                action = torch.tanh(action)
                return action.cpu().numpy()[0], mean.cpu().numpy()[0], log_std.cpu().numpy()[0]
    
    def get_action_log_prob(self, state: np.ndarray):
        if self.framework == 'tensorflow':
            if len(state.shape) == 1:
                state = np.expand_dims(state, axis=0)
            
            mean, log_std = self.model(state)
            log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
            std = tf.exp(log_std)
            
            normal = tfp.distributions.Normal(mean, std)
            x = normal.sample()
            action = tf.tanh(x)
            
            log_prob = normal.log_prob(x)
            log_prob -= tf.reduce_sum(tf.math.log(1 - action**2 + 1e-6), axis=1, keepdims=True)
            
            return action.numpy()[0], log_prob.numpy()[0]
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            mean, log_std = self.model(state_tensor)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = torch.exp(log_std)
            
            normal = Normal(mean, std)
            x = normal.rsample()
            action = torch.tanh(x)
            
            log_prob = normal.log_prob(x)
            log_prob -= torch.sum(torch.log(1 - action.pow(2) + 1e-6), dim=1, keepdim=True)
            
            return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]
    
    def save(self, filepath: str):
        if self.framework == 'tensorflow':
            self.model.save_weights(filepath)
        else:
            torch.save(self.model.state_dict(), filepath)
    
    def load(self, filepath: str):
        if self.framework == 'tensorflow':
            self.model.load_weights(filepath)
        else:
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
