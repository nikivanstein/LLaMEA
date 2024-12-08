import numpy as np

class DynamicNeighborhoodOptimizer:
    def __init__(self, budget, dim, num_neighbors=5, global_explore_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.num_neighbors = num_neighbors
        self.global_explore_rate = global_explore_rate

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.num_neighbors, self.dim))
        best_position = swarm[np.argmin([func(p) for p in swarm])]
        
        for _ in range(self.budget):
            for i in range(self.num_neighbors):
                neighbors = np.delete(swarm, i, axis=0)
                best_neighbor = neighbors[np.argmin([func(p) for p in neighbors])]
                candidate_position = np.clip(swarm[i] + np.random.normal(0, 0.1, self.dim), -5.0, 5.0)
                if func(candidate_position) < func(swarm[i]):
                    swarm[i] = candidate_position
                    if func(candidate_position) < func(best_position):
                        best_position = candidate_position
        return best_position