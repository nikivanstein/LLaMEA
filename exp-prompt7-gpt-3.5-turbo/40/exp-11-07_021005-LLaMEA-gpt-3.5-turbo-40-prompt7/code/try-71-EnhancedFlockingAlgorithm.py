import numpy as np

class EnhancedFlockingAlgorithm:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.population_size, self.max_velocity, self.c1, self.c2, self.w = budget, dim, 20, 0.1, 2.0, 2.0, 0.5
        self.positions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-self.max_velocity, self.max_velocity, (self.population_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_value = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            fitness_values = func(self.positions)
            update_personal = fitness_values < self.personal_best_values
            update_global = fitness_values < self.global_best_value
            self.personal_best_values[update_personal] = fitness_values[update_personal]
            self.personal_best_positions[update_personal] = self.positions[update_personal]
            self.global_best_value = np.where(update_global, fitness_values, self.global_best_value)
            self.global_best_position = np.where(update_global[:, np.newaxis], self.positions, self.global_best_position)

            r1, r2 = np.random.random((self.population_size, 1)), np.random.random((self.population_size, 1))
            self.velocities = self.w * self.velocities + self.c1 * r1 * (self.personal_best_positions - self.positions) + self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = np.clip(self.velocities, -self.max_velocity, self.max_velocity)
            self.positions = np.clip(self.positions + self.velocities, -5.0, 5.0)

        return self.global_best_value