import numpy as np

class OptimizedFlockingBirdsOptimization:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.population_size, self.max_velocity, self.c1, self.c2 = budget, dim, 20, 0.1, 2.0, 2.0
        self.positions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-self.max_velocity, self.max_velocity, (self.population_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_value = np.inf
        self.inertia_weight = 0.5

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                fitness = func(self.positions[i])
                if fitness < self.personal_best_values[i]:
                    self.personal_best_values[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = self.positions[i]

                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocities[i] = self.inertia_weight * self.velocities[i] + self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) + self.c2 * r2 * (self.global_best_position - self.positions[i])
                np.clip(self.velocities[i], -self.max_velocity, self.max_velocity, out=self.velocities[i])
                np.clip(self.positions[i] + self.velocities[i], -5.0, 5.0, out=self.positions[i])

            self.inertia_weight = max(0.4, self.inertia_weight - 0.01)

        return self.global_best_value