import numpy as np

class DynamicSwarmSizeOptimization:
    def __init__(self, budget, dim, alpha=0.1, beta=2.0):
        self.budget = budget
        self.dim = dim
        self.alpha = alpha
        self.beta = beta

    def __call__(self, func):
        swarm_size = max(5, min(20, self.budget // 3))  # Adjust swarm size dynamically based on budget
        swarm = np.random.uniform(-5.0, 5.0, (swarm_size, self.dim))
        costs = np.array([func(x) for x in swarm])

        for _ in range(self.budget // swarm_size):
            new_pos = swarm + self.alpha * np.mean(swarm, axis=0) + self.beta * np.random.uniform(-1, 1, (swarm_size, self.dim))
            new_pos = np.clip(new_pos, -5.0, 5.0)
            new_costs = np.array([func(x) for x in new_pos])

            improved_indices = np.where(new_costs < costs)[0]
            swarm[improved_indices] = new_pos[improved_indices]
            costs[improved_indices] = new_costs[improved_indices]

        best_idx = np.argmin(costs)
        best_solution = swarm[best_idx]
        return best_solution