import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ParallelizedEnhancedGradientSpiderSwarmOptimization:
    def __init__(self, budget, dim, swarm_size=10, alpha=0.1, beta=2.0, learning_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate

    def _evaluate_swarm(self, func, swarm):
        return np.array([func(x) for x in swarm])

    def _update_swarm(self, swarm, costs, gradients):
        new_pos = swarm + self.alpha * gradients + self.beta * np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        new_pos = np.clip(new_pos, -5.0, 5.0)
        new_costs = self._evaluate_swarm(func, new_pos)
        improved_indices = np.where(new_costs < costs)[0]
        swarm[improved_indices] = new_pos[improved_indices]
        costs[improved_indices] = new_costs[improved_indices]
        return swarm, costs

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        costs = self._evaluate_swarm(func, swarm)
        
        with ThreadPoolExecutor() as executor:
            for _ in range(self.budget // self.swarm_size):
                gradients = np.mean(swarm, axis=0) - swarm
                swarm, costs = self._update_swarm(swarm, costs, gradients)
        
        best_idx = np.argmin(costs)
        best_solution = swarm[best_idx]
        return best_solution