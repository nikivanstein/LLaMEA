import numpy as np
import concurrent.futures

class ParallelEnhancedSpiderSwarmOptimization:
    def __init__(self, budget, dim, swarm_size=10, alpha=0.1, beta=2.0):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        costs = np.array([func(x) for x in swarm])

        def evaluate_func(position):
            return func(position)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for _ in range(self.budget // self.swarm_size):
                new_pos = swarm + self.alpha * np.mean(swarm, axis=0) + self.beta * np.random.uniform(-1, 1, (self.swarm_size, self.dim))
                new_pos = np.clip(new_pos, -5.0, 5.0)
                new_costs = np.array(list(executor.map(evaluate_func, new_pos)))

                improved_indices = np.where(new_costs < costs)[0]
                swarm[improved_indices] = new_pos[improved_indices]
                costs[improved_indices] = new_costs[improved_indices]

        best_idx = np.argmin(costs)
        best_solution = swarm[best_idx]
        return best_solution