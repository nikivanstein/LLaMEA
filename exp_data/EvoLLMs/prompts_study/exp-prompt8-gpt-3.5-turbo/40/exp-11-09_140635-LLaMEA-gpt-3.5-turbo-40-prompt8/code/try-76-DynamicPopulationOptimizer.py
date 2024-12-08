import numpy as np

class DynamicPopulationOptimizer:
    def __init__(self, budget, dim, num_neighbors=5):
        self.budget = budget
        self.dim = dim
        self.num_neighbors = num_neighbors

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.num_neighbors, self.dim))
        best_position = swarm[np.argmin([func(p) for p in swarm])]
        mutation_rate = 1.0
        
        for _ in range(self.budget):
            for i in range(len(swarm)):
                candidate_position = np.clip(swarm[i] + np.random.normal(0, mutation_rate, self.dim), -5.0, 5.0)
                if func(candidate_position) < func(swarm[i]):
                    swarm[i] = candidate_position
                    if func(candidate_position) < func(best_position):
                        best_position = candidate_position
                        mutation_rate *= 0.99  # Adapt mutation rate based on performance
            
            if np.random.rand() < 0.1:  # Dynamic Population Resizing
                if np.random.rand() < 0.5 and len(swarm) < 20:
                    swarm = np.vstack([swarm, np.random.uniform(-5.0, 5.0, (1, self.dim))])
                elif len(swarm) > 1:
                    remove_idx = np.random.choice(len(swarm))
                    swarm = np.delete(swarm, remove_idx, axis=0)
        
        return best_position