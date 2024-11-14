import numpy as np

class EnhancedInertiaOptimizer:
    def __init__(self, budget, dim, num_neighbors=5, inertia_weight=0.5, inertia_decay=0.95):
        self.budget = budget
        self.dim = dim
        self.num_neighbors = num_neighbors
        self.inertia_weight = inertia_weight
        self.inertia_decay = inertia_decay

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.num_neighbors, self.dim))
        best_position = swarm[np.argmin([func(p) for p in swarm])
        inertia_weight = self.inertia_weight

        for _ in range(self.budget):
            for i in range(self.num_neighbors):
                candidate_position = np.clip(swarm[i] + np.random.normal(0, 1, self.dim), -5.0, 5.0)
                if func(candidate_position) < func(swarm[i]):
                    swarm[i] = candidate_position
                    if func(candidate_position) < func(best_position):
                        best_position = candidate_position
                        inertia_weight = max(0.2, inertia_weight * self.inertia_decay)  # Adapt inertia weight based on fitness improvement
                else:
                    swarm[i] = swarm[i] + inertia_weight * (candidate_position - swarm[i])

        return best_position