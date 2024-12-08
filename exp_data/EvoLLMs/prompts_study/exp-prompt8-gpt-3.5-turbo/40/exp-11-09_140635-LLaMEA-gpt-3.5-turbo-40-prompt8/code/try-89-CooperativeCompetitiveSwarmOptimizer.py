import numpy as np

class CooperativeCompetitiveSwarmOptimizer:
    def __init__(self, budget, dim, num_neighbors=5, global_explore_rate=0.1, competitive_rate=0.2):
        self.budget = budget
        self.dim = dim
        self.num_neighbors = num_neighbors
        self.global_explore_rate = global_explore_rate
        self.competitive_rate = competitive_rate

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.num_neighbors, self.dim))
        best_position = swarm[np.argmin([func(p) for p in swarm])]
        mutation_rate = 1.0

        for _ in range(self.budget):
            for i in range(self.num_neighbors):
                if np.random.rand() < self.global_explore_rate:
                    candidate_position = np.random.uniform(-5.0, 5.0, self.dim)
                else:
                    candidate_position = np.clip(swarm[i] + np.random.normal(0, mutation_rate, self.dim), -5.0, 5.0)
                if func(candidate_position) < func(swarm[i]):
                    swarm[i] = candidate_position
                    if np.random.rand() < self.competitive_rate:
                        competition_indices = np.random.choice(np.delete(np.arange(self.num_neighbors), i), 2, replace=False)
                        competitor = np.argmin([func(swarm[c]) for c in competition_indices])
                        if func(swarm[competition_indices[competitor]]) < func(best_position):
                            best_position = swarm[competition_indices[competitor]]
                if func(candidate_position) < func(best_position):
                    best_position = candidate_position
                    mutation_rate *= 0.99  # Adapt mutation rate based on performance
        return best_position