import numpy as np

class EnhancedFlockingBirdsOptimization:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.population_size, self.max_velocity, self.c1, self.c2, self.w = budget, dim, 20, 0.1, 2.0, 2.0, 0.5
        self.positions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-self.max_velocity, self.max_velocity, (self.population_size, self.dim))
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_value = np.inf

    def __call__(self, func):
        personal_best_positions = self.positions.copy()
        global_best_position = np.zeros(self.dim)

        for _ in range(self.budget):
            fitness_values = np.apply_along_axis(func, 1, self.positions)

            update_personal_best = fitness_values < self.personal_best_values
            self.personal_best_values = np.where(update_personal_best, fitness_values, self.personal_best_values)
            personal_best_positions = np.where(update_personal_best[:, None], self.positions, personal_best_positions)

            best_idx = np.argmin(self.personal_best_values)
            if self.personal_best_values[best_idx] < self.global_best_value:
                self.global_best_value = self.personal_best_values[best_idx]
                global_best_position = self.positions[best_idx]

            r1, r2 = np.random.random((self.population_size, self.dim)), np.random.random((self.population_size, self.dim))
            self.velocities = self.w * self.velocities + self.c1 * r1 * (personal_best_positions - self.positions) + self.c2 * r2 * (global_best_position - self.positions)
            self.velocities = np.clip(self.velocities, -self.max_velocity, self.max_velocity)
            self.positions = np.clip(self.positions + self.velocities, -5.0, 5.0)

        return self.global_best_value