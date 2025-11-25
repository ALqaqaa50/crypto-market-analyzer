import numpy as np
import logging

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


class CriticNetwork:
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [512, 256, 128],
                 learning_rate: float = 0.001, use_gpu: bool = True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        
        if HAS_TF:
            self.framework = 'tensorflow'
            self._build_tf_model()
        elif HAS_TORCH:
            self.framework = 'pytorch'
            self._build_torch_model()
        else:
            raise ImportError("Neither TensorFlow nor PyTorch available")
        
        logger.info(f"CriticNetwork initialized: framework={self.framework}")
    
    def _build_tf_model(self):
        state_input = keras.Input(shape=(self.state_dim,))
        action_input = keras.Input(shape=(self.action_dim,))
        
        concat = layers.Concatenate()([state_input, action_input])
        x = concat
        
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='relu', kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        output = layers.Dense(1, kernel_initializer=keras.initializers.RandomUniform(-0.003, 0.003))(x)
        
        self.model = keras.Model(inputs=[state_input, action_input], outputs=output)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
    
    def _build_torch_model(self):
        class TorchCritic(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dims):
                super(TorchCritic, self).__init__()
                
                input_dim = state_dim + action_dim
                layers_list = []
                prev_dim = input_dim
                
                for dim in hidden_dims:
                    layers_list.append(nn.Linear(prev_dim, dim))
                    layers_list.append(nn.ReLU())
                    layers_list.append(nn.BatchNorm1d(dim))
                    layers_list.append(nn.Dropout(0.2))
                    prev_dim = dim
                
                layers_list.append(nn.Linear(prev_dim, 1))
                
                self.network = nn.Sequential(*layers_list)
            
            def forward(self, state, action):
                x = torch.cat([state, action], dim=1)
                return self.network(x)
        
        self.model = TorchCritic(self.state_dim, self.action_dim, self.hidden_dims)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        if self.use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        if self.framework == 'tensorflow':
            if len(state.shape) == 1:
                state = np.expand_dims(state, axis=0)
            if len(action.shape) == 1:
                action = np.expand_dims(action, axis=0)
            
            q_value = self.model([state, action], training=False).numpy()
            return q_value[0] if q_value.shape[0] == 1 else q_value
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
            action_tensor = torch.FloatTensor(action).to(self.device)
            
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            if len(action_tensor.shape) == 1:
                action_tensor = action_tensor.unsqueeze(0)
            
            self.model.eval()
            with torch.no_grad():
                q_value = self.model(state_tensor, action_tensor).cpu().numpy()
            return q_value[0] if q_value.shape[0] == 1 else q_value
    
    def train_step(self, states: np.ndarray, actions: np.ndarray, target_q: np.ndarray):
        if self.framework == 'tensorflow':
            with tf.GradientTape() as tape:
                q_values = self.model([states, actions], training=True)
                loss = tf.reduce_mean(tf.square(q_values - target_q))
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return float(loss.numpy())
        else:
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions_tensor = torch.FloatTensor(actions).to(self.device)
            target_q_tensor = torch.FloatTensor(target_q).to(self.device)
            
            self.model.train()
            self.optimizer.zero_grad()
            
            q_values = self.model(states_tensor, actions_tensor)
            loss = F.mse_loss(q_values, target_q_tensor)
            
            loss.backward()
            self.optimizer.step()
            
            return float(loss.item())
    
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
    
    def get_weights(self):
        if self.framework == 'tensorflow':
            return self.model.get_weights()
        else:
            return [param.data.cpu().numpy() for param in self.model.parameters()]
    
    def set_weights(self, weights):
        if self.framework == 'tensorflow':
            self.model.set_weights(weights)
        else:
            for param, weight in zip(self.model.parameters(), weights):
                param.data = torch.FloatTensor(weight).to(self.device)
