import numpy as np
import logging
from typing import Dict, List
from okx_stream_hunter.ai.rl_v2.agents.ddpg_agent import DDPGAgent
from okx_stream_hunter.ai.rl_v2.agents.td3_agent import TD3Agent
from okx_stream_hunter.ai.rl_v2.agents.sac_agent import SACAgent

logger = logging.getLogger(__name__)


class MultiAgentRL:
    def __init__(self, state_dim: int, action_dim: int, num_agents: int = 3,
                 agent_types: List[str] = ['ddpg', 'td3', 'sac'],
                 hidden_dims_actor: list = [512, 256, 128],
                 hidden_dims_critic: list = [512, 256, 128],
                 use_gpu: bool = True):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.agent_types = agent_types
        
        self.agents = []
        self.agent_names = []
        self.agent_weights = np.ones(num_agents) / num_agents
        self.agent_performance = np.zeros(num_agents)
        self.agent_episode_count = np.zeros(num_agents)
        
        for i, agent_type in enumerate(agent_types[:num_agents]):
            if agent_type.lower() == 'ddpg':
                agent = DDPGAgent(state_dim, action_dim, hidden_dims_actor, hidden_dims_critic,
                                 use_gpu=use_gpu)
                name = f"DDPG_{i}"
            elif agent_type.lower() == 'td3':
                agent = TD3Agent(state_dim, action_dim, hidden_dims_actor, hidden_dims_critic,
                                use_gpu=use_gpu)
                name = f"TD3_{i}"
            elif agent_type.lower() == 'sac':
                agent = SACAgent(state_dim, action_dim, hidden_dims_actor[:2], hidden_dims_critic,
                                use_gpu=use_gpu)
                name = f"SAC_{i}"
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            self.agents.append(agent)
            self.agent_names.append(name)
        
        self.active_agent_idx = 0
        self.training_steps = 0
        
        logger.info(f"MultiAgentRL initialized with {num_agents} agents: {self.agent_names}")
    
    def select_action(self, state: np.ndarray, training: bool = True, 
                     use_voting: bool = False) -> np.ndarray:
        if use_voting:
            actions = []
            for agent in self.agents:
                if isinstance(agent, SACAgent):
                    action = agent.select_action(state, deterministic=not training)
                else:
                    action = agent.select_action(state, training=training)
                actions.append(action)
            
            weighted_action = np.average(actions, axis=0, weights=self.agent_weights)
            return weighted_action
        else:
            agent = self.agents[self.active_agent_idx]
            if isinstance(agent, SACAgent):
                return agent.select_action(state, deterministic=not training)
            else:
                return agent.select_action(state, training=training)
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool, agent_idx: int = None):
        if agent_idx is None:
            agent_idx = self.active_agent_idx
        
        self.agents[agent_idx].store_transition(state, action, reward, next_state, done)
        
        if done:
            self.agent_performance[agent_idx] += reward
            self.agent_episode_count[agent_idx] += 1
    
    def train(self, train_all: bool = True) -> Dict:
        results = {}
        
        if train_all:
            for i, agent in enumerate(self.agents):
                agent_result = agent.train()
                if agent_result:
                    results[self.agent_names[i]] = agent_result
        else:
            agent = self.agents[self.active_agent_idx]
            agent_result = agent.train()
            if agent_result:
                results[self.agent_names[self.active_agent_idx]] = agent_result
        
        self.training_steps += 1
        
        if self.training_steps % 1000 == 0:
            self._update_agent_weights()
        
        return results
    
    def _update_agent_weights(self):
        avg_performance = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            if self.agent_episode_count[i] > 0:
                avg_performance[i] = self.agent_performance[i] / self.agent_episode_count[i]
        
        if avg_performance.sum() > 0:
            self.agent_weights = np.exp(avg_performance) / np.exp(avg_performance).sum()
        else:
            self.agent_weights = np.ones(self.num_agents) / self.num_agents
        
        best_idx = np.argmax(avg_performance)
        self.active_agent_idx = best_idx
        
        logger.info(f"Agent weights updated: {dict(zip(self.agent_names, self.agent_weights))}")
        logger.info(f"Active agent switched to: {self.agent_names[best_idx]}")
    
    def get_best_agent(self) -> tuple:
        avg_performance = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            if self.agent_episode_count[i] > 0:
                avg_performance[i] = self.agent_performance[i] / self.agent_episode_count[i]
        
        best_idx = np.argmax(avg_performance)
        return self.agents[best_idx], self.agent_names[best_idx], avg_performance[best_idx]
    
    def get_agent_stats(self) -> Dict:
        stats = {}
        for i, name in enumerate(self.agent_names):
            avg_perf = 0
            if self.agent_episode_count[i] > 0:
                avg_perf = self.agent_performance[i] / self.agent_episode_count[i]
            
            stats[name] = {
                'weight': float(self.agent_weights[i]),
                'total_performance': float(self.agent_performance[i]),
                'episode_count': int(self.agent_episode_count[i]),
                'avg_performance': float(avg_perf),
                'is_active': i == self.active_agent_idx
            }
        
        return stats
    
    def save(self, filepath_prefix: str):
        for i, agent in enumerate(self.agents):
            agent.save(f"{filepath_prefix}_{self.agent_names[i]}")
        
        np.savez(f"{filepath_prefix}_multi_agent_state.npz",
                agent_weights=self.agent_weights,
                agent_performance=self.agent_performance,
                agent_episode_count=self.agent_episode_count,
                active_agent_idx=self.active_agent_idx,
                training_steps=self.training_steps)
        
        logger.info(f"MultiAgentRL saved: {filepath_prefix}")
    
    def load(self, filepath_prefix: str):
        for i, agent in enumerate(self.agents):
            try:
                agent.load(f"{filepath_prefix}_{self.agent_names[i]}")
            except Exception as e:
                logger.warning(f"Failed to load {self.agent_names[i]}: {e}")
        
        try:
            state = np.load(f"{filepath_prefix}_multi_agent_state.npz")
            self.agent_weights = state['agent_weights']
            self.agent_performance = state['agent_performance']
            self.agent_episode_count = state['agent_episode_count']
            self.active_agent_idx = int(state['active_agent_idx'])
            self.training_steps = int(state['training_steps'])
        except Exception as e:
            logger.warning(f"Failed to load multi-agent state: {e}")
        
        logger.info(f"MultiAgentRL loaded: {filepath_prefix}")
