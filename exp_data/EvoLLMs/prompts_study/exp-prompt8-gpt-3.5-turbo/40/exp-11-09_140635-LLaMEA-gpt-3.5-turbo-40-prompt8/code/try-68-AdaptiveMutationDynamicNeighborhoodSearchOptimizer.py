import numpy as np

class AdaptiveMutationDynamicNeighborhoodSearchOptimizer:
    def __init__(self, budget, dim, num_neighbors=5, initial_mut_prob=0.5):
        self.budget = budget
        self.dim = dim
        self.num_neighbors = num_neighbors
        self.mut_prob = initial_mut_prob

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.num_neighbors, self.dim))
        best_position = swarm[np.argmin([func(p) for p in swarm])
        diversity_history = [np.mean(np.std(swarm, axis=0))]

        for _ in range(self.budget):
            for i in range(self.num_neighbors):
                candidate_position = np.clip(swarm[i] + np.random.normal(0, 1, self.dim) * self.mut_prob, -5.0, 5.0)
                if func(candidate_position) < func(swarm[i]):
                    swarm[i] = candidate_position
                    if func(candidate_position) < func(best_position):
                        best_position = candidate_position
            diversity = np.mean(np.std(swarm, axis=0))
            if diversity > diversity_history[-1]:
                self.mut_prob *= 1.05  # Increase mutation probability
            else:
                self.mut_prob *= 0.95  # Decrease mutation probability
            diversity_history.append(diversity)

        return best_position