import numpy as np

class EnhancedMutationOptimizer:
    def __init__(self, budget, dim, num_neighbors=5, global_explore_rate=0.1, adapt_rate=0.9):
        self.budget = budget
        self.dim = dim
        self.num_neighbors = num_neighbors
        self.global_explore_rate = global_explore_rate
        self.adapt_rate = adapt_rate

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.num_neighbors, self.dim))
        best_position = swarm[np.argmin([func(p) for p in swarm])
        mutation_rate = 1.0

        for _ in range(self.budget):
            for i in range(self.num_neighbors):
                if np.random.rand() < self.global_explore_rate:
                    candidate_position = np.random.uniform(-5.0, 5.0, self.dim)
                else:
                    candidate_position = np.clip(swarm[i] + np.random.normal(0, mutation_rate, self.dim), -5.0, 5.0)
                if func(candidate_position) < func(swarm[i]):
                    swarm[i] = candidate_position
                    if func(candidate_position) < func(best_position):
                        best_position = candidate_position
                        self.global_explore_rate *= self.adapt_rate  # Adjust global exploration rate based on performance
        return best_position