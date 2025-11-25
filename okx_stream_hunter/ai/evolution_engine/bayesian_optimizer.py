import numpy as np
import logging
from typing import Dict, List, Callable
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


class BayesianHyperparameterOptimizer:
    def __init__(self, param_bounds: Dict[str, tuple], n_iter: int = 100,
                 n_init: int = 10, acquisition: str = 'ei', xi: float = 0.01,
                 kappa: float = 2.576, random_state: int = 42):
        
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_iter = n_iter
        self.n_init = n_init
        self.acquisition = acquisition
        self.xi = xi
        self.kappa = kappa
        self.random_state = random_state
        
        self.X_observed = []
        self.y_observed = []
        
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=random_state,
            normalize_y=True
        )
        
        self.best_params = None
        self.best_value = -np.inf
        self.iteration = 0
        
        logger.info(f"BayesianHyperparameterOptimizer initialized: n_iter={n_iter}, "
                   f"acquisition={acquisition}, params={self.param_names}")
    
    def _normalize_params(self, params: np.ndarray) -> Dict:
        param_dict = {}
        for i, name in enumerate(self.param_names):
            lower, upper = self.param_bounds[name]
            value = params[i] * (upper - lower) + lower
            
            if isinstance(lower, int) and isinstance(upper, int):
                param_dict[name] = int(round(value))
            else:
                param_dict[name] = float(value)
        
        return param_dict
    
    def _denormalize_params(self, param_dict: Dict) -> np.ndarray:
        params = np.zeros(len(self.param_names))
        for i, name in enumerate(self.param_names):
            lower, upper = self.param_bounds[name]
            value = param_dict[name]
            params[i] = (value - lower) / (upper - lower)
        
        return params
    
    def _sample_random_params(self) -> np.ndarray:
        return np.random.uniform(0, 1, size=len(self.param_names))
    
    def _expected_improvement(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(-1, len(self.param_names))
        
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        mu_best = np.max(self.y_observed)
        
        with np.errstate(divide='warn'):
            Z = (mu - mu_best - self.xi) / sigma
            ei = (mu - mu_best - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return -ei.flatten()
    
    def _upper_confidence_bound(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(-1, len(self.param_names))
        
        mu, sigma = self.gp.predict(X, return_std=True)
        
        ucb = mu + self.kappa * sigma
        
        return -ucb.flatten()
    
    def _probability_of_improvement(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(-1, len(self.param_names))
        
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)
        
        mu_best = np.max(self.y_observed)
        
        with np.errstate(divide='warn'):
            Z = (mu - mu_best - self.xi) / sigma
            pi = norm.cdf(Z)
            pi[sigma == 0.0] = 0.0
        
        return -pi.flatten()
    
    def _get_acquisition_func(self):
        if self.acquisition == 'ei':
            return self._expected_improvement
        elif self.acquisition == 'ucb':
            return self._upper_confidence_bound
        elif self.acquisition == 'poi':
            return self._probability_of_improvement
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition}")
    
    def _propose_next_params(self) -> np.ndarray:
        acquisition_func = self._get_acquisition_func()
        
        best_x = None
        best_acquisition = np.inf
        
        n_restarts = 25
        for _ in range(n_restarts):
            x0 = self._sample_random_params()
            
            result = minimize(
                acquisition_func,
                x0,
                bounds=[(0, 1)] * len(self.param_names),
                method='L-BFGS-B'
            )
            
            if result.fun < best_acquisition:
                best_acquisition = result.fun
                best_x = result.x
        
        return best_x
    
    def optimize(self, objective_func: Callable[[Dict], float]) -> Dict:
        for i in range(self.n_iter):
            self.iteration = i
            
            if i < self.n_init:
                next_params = self._sample_random_params()
            else:
                self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
                next_params = self._propose_next_params()
            
            param_dict = self._normalize_params(next_params)
            
            try:
                value = objective_func(param_dict)
            except Exception as e:
                logger.error(f"Error evaluating params: {e}")
                value = -np.inf
            
            self.X_observed.append(next_params)
            self.y_observed.append(value)
            
            if value > self.best_value:
                self.best_value = value
                self.best_params = param_dict
            
            logger.info(f"Iteration {i+1}/{self.n_iter}: Value={value:.4f}, Best={self.best_value:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'iterations': self.n_iter,
            'all_params': [self._normalize_params(x) for x in self.X_observed],
            'all_values': self.y_observed
        }
    
    def get_best(self) -> tuple:
        return self.best_params, self.best_value
    
    def suggest_next(self) -> Dict:
        if len(self.X_observed) < self.n_init:
            next_params = self._sample_random_params()
        else:
            self.gp.fit(np.array(self.X_observed), np.array(self.y_observed))
            next_params = self._propose_next_params()
        
        return self._normalize_params(next_params)
    
    def report(self, params: Dict, value: float):
        denorm_params = self._denormalize_params(params)
        self.X_observed.append(denorm_params)
        self.y_observed.append(value)
        
        if value > self.best_value:
            self.best_value = value
            self.best_params = params
    
    def save_results(self, filepath: str):
        import json
        results = {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'param_bounds': self.param_bounds,
            'all_params': [self._normalize_params(x) for x in self.X_observed],
            'all_values': self.y_observed,
            'iteration': self.iteration
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved: {filepath}")
