import numpy as np

class AdaptiveMutationOptimizer:
    def __init__(self, budget, dim, num_neighbors=5):
        self.budget = budget
        self.dim = dim
        self.num_neighbors = num_neighbors

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.num_neighbors, self.dim))
        best_position = swarm[np.argmin([func(p) for p in swarm])]
        mutation_rate = 1.0

        for _ in range(self.budget):
            for i in range(self.num_neighbors):
                candidate_position = np.clip(swarm[i] + np.random.normal(0, mutation_rate, self.dim), -5.0, 5.0)
                if func(candidate_position) < func(swarm[i]):
                    swarm[i] = candidate_position
                    if func(candidate_position) < func(best_position):
                        best_position = candidate_position
                        mutation_rate *= 0.99  # Adapt mutation rate based on performance
        return best_position