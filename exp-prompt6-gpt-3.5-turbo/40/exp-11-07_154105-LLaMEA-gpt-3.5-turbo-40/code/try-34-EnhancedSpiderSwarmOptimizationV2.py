import numpy as np

class EnhancedSpiderSwarmOptimizationV2:
    def __init__(self, budget, dim, swarm_size=10, alpha=0.1, beta=2.0):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        costs = np.array([func(x) for x in swarm])
        evals_per_iteration = self.budget // self.swarm_size  # Reduce loop complexity
        
        for _ in range(evals_per_iteration):
            new_pos = swarm + self.alpha * np.mean(swarm, axis=0) + self.beta * np.random.uniform(-1, 1, (self.swarm_size, self.dim))
            new_pos = np.clip(new_pos, -5.0, 5.0)
            new_costs = np.array([func(x) for x in new_pos])
            
            improved_mask = new_costs < costs
            swarm[improved_mask] = new_pos[improved_mask]
            costs[improved_mask] = new_costs[improved_mask]
        
        best_idx = np.argmin(costs)
        best_solution = swarm[best_idx]
        return best_solution