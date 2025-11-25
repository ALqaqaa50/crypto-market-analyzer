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


class ActorNetwork:
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [512, 256, 128],
                 learning_rate: float = 0.0003, use_gpu: bool = True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu and (HAS_TF or HAS_TORCH)
        
        if HAS_TF:
            self.framework = 'tensorflow'
            self._build_tf_model()
        elif HAS_TORCH:
            self.framework = 'pytorch'
            self._build_torch_model()
        else:
            raise ImportError("Neither TensorFlow nor PyTorch available")
        
        logger.info(f"ActorNetwork initialized: framework={self.framework}, state_dim={state_dim}, "
                   f"action_dim={action_dim}, hidden_dims={hidden_dims}")
    
    def _build_tf_model(self):
        inputs = keras.Input(shape=(self.state_dim,))
        x = inputs
        
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='relu', kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(self.action_dim, activation='tanh', 
                              kernel_initializer=keras.initializers.RandomUniform(-0.003, 0.003))(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        if self.use_gpu:
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass
    
    def _build_torch_model(self):
        class TorchActor(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dims):
                super(TorchActor, self).__init__()
                
                layers_list = []
                prev_dim = state_dim
                for dim in hidden_dims:
                    layers_list.append(nn.Linear(prev_dim, dim))
                    layers_list.append(nn.ReLU())
                    layers_list.append(nn.BatchNorm1d(dim))
                    layers_list.append(nn.Dropout(0.2))
                    prev_dim = dim
                
                layers_list.append(nn.Linear(prev_dim, action_dim))
                layers_list.append(nn.Tanh())
                
                self.network = nn.Sequential(*layers_list)
            
            def forward(self, state):
                return self.network(state)
        
        self.model = TorchActor(self.state_dim, self.action_dim, self.hidden_dims)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        if self.use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    def predict(self, state: np.ndarray, training: bool = False) -> np.ndarray:
        if self.framework == 'tensorflow':
            if len(state.shape) == 1:
                state = np.expand_dims(state, axis=0)
            action = self.model(state, training=training).numpy()
            return action[0] if action.shape[0] == 1 else action
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            if training:
                self.model.train()
            else:
                self.model.eval()
            
            with torch.no_grad():
                action = self.model(state_tensor).cpu().numpy()
            return action[0] if action.shape[0] == 1 else action
    
    def train_step(self, states: np.ndarray, critic_gradients: np.ndarray):
        if self.framework == 'tensorflow':
            with tf.GradientTape() as tape:
                actions = self.model(states, training=True)
                actor_loss = -tf.reduce_mean(actions * critic_gradients)
            
            gradients = tape.gradient(actor_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return float(actor_loss.numpy())
        else:
            states_tensor = torch.FloatTensor(states).to(self.device)
            critic_gradients_tensor = torch.FloatTensor(critic_gradients).to(self.device)
            
            self.model.train()
            self.optimizer.zero_grad()
            
            actions = self.model(states_tensor)
            actor_loss = -torch.mean(actions * critic_gradients_tensor)
            
            actor_loss.backward()
            self.optimizer.step()
            
            return float(actor_loss.item())
    
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
