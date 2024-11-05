import numpy as np

class ModifiedHybridParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 30
        self.population_size = self.initial_population_size
        self.inertia_weight = 0.7
        self.cognitive_component = 1.5
        self.social_component = 2.0
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def sigmoid_warp(self, v):
        return 1 / (1 + np.exp(-v))

    def update_population_size(self):
        # Adaptive population control with a more aggressive reduction
        new_size = max(5, int(self.initial_population_size * np.sqrt(1 - self.evaluations / self.budget)))
        if new_size < self.population_size:
            indices = np.argsort(self.personal_best_scores)[:new_size]
            self.positions = self.positions[indices]
            self.velocities = self.velocities[indices]
            self.personal_best_positions = self.personal_best_positions[indices]
            self.personal_best_scores = self.personal_best_scores[indices]
            self.population_size = new_size

    def opposition_based_learning(self, position):
        return self.lower_bound + self.upper_bound - position

    def __call__(self, func):
        while self.evaluations < self.budget:
            self.update_population_size()
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                fitness = func(self.positions[i])
                self.evaluations += 1

                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i]

            inertia_weight = self.inertia_weight * (self.budget - self.evaluations) / self.budget

            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                cognitive_velocity = self.cognitive_component * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_component * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                self.velocities[i] = self.sigmoid_warp(self.velocities[i])  # Sigmoid warp
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Implement Opposition-Based Learning
            if np.random.rand() < 0.1:
                for i in range(self.population_size):
                    opposite_position = self.opposition_based_learning(self.positions[i])
                    opposite_fitness = func(opposite_position)
                    self.evaluations += 1
                    if opposite_fitness < self.personal_best_scores[i]:
                        self.positions[i] = opposite_position
                        self.personal_best_scores[i] = opposite_fitness
                        self.personal_best_positions[i] = opposite_position

        return self.global_best_score