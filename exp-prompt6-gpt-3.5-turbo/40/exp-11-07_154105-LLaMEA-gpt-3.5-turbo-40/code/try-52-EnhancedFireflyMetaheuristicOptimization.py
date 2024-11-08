import numpy as np

class EnhancedFireflyMetaheuristicOptimization:
    def __init__(self, budget, dim, swarm_size=10, alpha=0.1, beta=2.0):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        costs = np.array([func(x) for x in swarm])
        
        for _ in range(self.budget // self.swarm_size):
            attractiveness = np.expand_dims(np.power(costs, -1), axis=1)
            movement = self.alpha * np.mean(swarm, axis=0) - swarm + self.beta * np.random.uniform(-1, 1, (self.swarm_size, self.dim))
            new_pos = swarm + attractiveness * movement
            new_pos = np.clip(new_pos, -5.0, 5.0)
            new_costs = np.array([func(x) for x in new_pos])
            
            improved_indices = np.where(new_costs < costs)[0]
            swarm[improved_indices] = new_pos[improved_indices]
            costs[improved_indices] = new_costs[improved_indices]
        
        best_idx = np.argmin(costs)
        best_solution = swarm[best_idx]
        return best_solution