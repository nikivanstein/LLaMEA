import numpy as np

class DynamicInertiaFlockingBirdsOptimization:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.population_size, self.max_velocity, self.c1, self.c2 = budget, dim, 20, 0.1, 2.0, 2.0
        self.positions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-self.max_velocity, self.max_velocity, (self.population_size, self.dim))
        self.personal_best_positions, self.personal_best_values = self.positions.copy(), np.full(self.population_size, np.inf)
        self.global_best_position, self.global_best_value = np.zeros(self.dim), np.inf
        self.w_min, self.w_max = 0.4, 0.9

    def __call__(self, func):
        for t in range(1, self.budget + 1):
            w = self.w_min + (self.w_max - self.w_min) * (self.budget - t) / self.budget
            for i in range(self.population_size):
                fitness = func(self.positions[i])
                if fitness < self.personal_best_values[i]:
                    self.personal_best_values[i], self.personal_best_positions[i] = fitness, self.positions[i]
                if fitness < self.global_best_value:
                    self.global_best_value, self.global_best_position = fitness, self.positions[i]

                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocities[i] = w * self.velocities[i] + self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) + self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = np.clip(self.velocities[i], -self.max_velocity, self.max_velocity)
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], -5.0, 5.0)

        return self.global_best_value