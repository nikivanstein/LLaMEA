import numpy as np

class ImprovedDynamicNeighborhoodSearchOptimizer:
    def __init__(self, budget, dim, num_neighbors=5, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.num_neighbors = num_neighbors
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.num_neighbors, self.dim))
        best_position = swarm[np.argmin([func(p) for p in swarm])]
        
        for _ in range(self.budget):
            for i in range(self.num_neighbors):
                mutation_strength = np.random.normal(0, 1, self.dim) * self.mutation_rate * np.abs(swarm[i] - best_position)
                candidate_position = np.clip(swarm[i] + mutation_strength, -5.0, 5.0)
                if func(candidate_position) < func(swarm[i]):
                    swarm[i] = candidate_position
                    if func(candidate_position) < func(best_position):
                        best_position = candidate_position
        return best_position