import numpy as np

class AdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-0.9, 0.9, (self.population_size, dim))  # Reduced initial velocity magnitude
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest_position = None
        self.gbest_score = float('inf')
        self.evaluations = 0
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5  # Slightly tweaked cognitive coefficient
        self.social_coeff = 1.4

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                fitness = func(self.positions[i])
                self.evaluations += 1

                if fitness < self.pbest_scores[i]:
                    self.pbest_scores[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()

                if fitness < self.gbest_score:
                    self.gbest_score = fitness
                    self.gbest_position = self.positions[i].copy()

            if np.random.rand() < 0.52:  # Adjusted exploration-exploitation threshold
                self.inertia_weight = 0.8
                self.cognitive_coeff = 1.7
                self.social_coeff = 1.2
            else:
                self.inertia_weight = 0.4
                self.cognitive_coeff = 1.3
                self.social_coeff = 1.9

            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_coeff * r1 * (self.pbest_positions[i] - self.positions[i])
                social_velocity = self.social_coeff * r2 * (self.gbest_position - self.positions[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)

                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.gbest_position, self.gbest_score