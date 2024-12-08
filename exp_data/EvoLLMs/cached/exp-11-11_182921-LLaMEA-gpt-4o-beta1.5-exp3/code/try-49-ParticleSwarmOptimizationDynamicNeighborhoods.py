import numpy as np

class ParticleSwarmOptimizationDynamicNeighborhoods:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.best_positions = np.copy(self.population)
        self.best_scores = np.full(self.population_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.func_evaluations = 0
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.neighborhood_size = max(1, self.population_size // 10)

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                current_score = func(self.population[i])
                self.func_evaluations += 1

                # Update personal best
                if current_score < self.best_scores[i]:
                    self.best_scores[i] = current_score
                    self.best_positions[i] = self.population[i].copy()

                # Update global best
                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.population[i].copy()

            # Dynamic neighborhood topology
            neighbors_indices = np.random.choice(self.population_size, self.neighborhood_size, replace=False)
            neighborhood_best_score = float('inf')
            neighborhood_best_position = None
            for neighbor_idx in neighbors_indices:
                if self.best_scores[neighbor_idx] < neighborhood_best_score:
                    neighborhood_best_score = self.best_scores[neighbor_idx]
                    neighborhood_best_position = self.best_positions[neighbor_idx]

            # Update velocities and positions
            for i in range(self.population_size):
                inertia = self.inertia_weight * self.velocities[i]
                cognitive = self.cognitive_coefficient * np.random.rand(self.dim) * (self.best_positions[i] - self.population[i])
                social = self.social_coefficient * np.random.rand(self.dim) * (neighborhood_best_position - self.population[i])
                self.velocities[i] = inertia + cognitive + social
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

        return self.global_best_position