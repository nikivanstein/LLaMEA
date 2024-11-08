import numpy as np

class EnhancedFlockingBirdsOptimizationImproved:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.population_size, self.max_velocity, self.c1, self.c2, self.w = budget, dim, 20, 0.1, 2.0, 2.0, 0.5
        self.positions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-self.max_velocity, self.max_velocity, (self.population_size, self.dim))
        self.personal_best_positions, self.personal_best_values = self.positions.copy(), np.full(self.population_size, np.inf)
        self.global_best_position, self.global_best_value = np.zeros(self.dim), np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = np.array([func(p) for p in self.positions])
            update_personal_best = fitness < self.personal_best_values
            self.personal_best_values[update_personal_best] = fitness[update_personal_best]
            self.personal_best_positions[update_personal_best] = self.positions[update_personal_best]

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.global_best_value:
                self.global_best_value, self.global_best_position = fitness[best_idx], self.positions[best_idx]

            r1, r2 = np.random.random((self.population_size, self.dim)), np.random.random((self.population_size, self.dim))
            self.velocities = self.w * self.velocities + self.c1 * r1 * (self.personal_best_positions - self.positions) + self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = np.clip(self.velocities, -self.max_velocity, self.max_velocity)
            self.positions = np.clip(self.positions + self.velocities, -5.0, 5.0)

        return self.global_best_value