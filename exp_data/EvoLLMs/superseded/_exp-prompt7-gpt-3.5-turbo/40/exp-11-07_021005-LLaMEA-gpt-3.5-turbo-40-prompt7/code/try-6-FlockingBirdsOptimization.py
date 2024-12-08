import numpy as np

class FlockingBirdsOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_velocity = 0.1
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.5
        self.positions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-self.max_velocity, self.max_velocity, (self.population_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_value = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.population_size):
                fitness = func(self.positions[i])
                if fitness < self.personal_best_values[i]:
                    self.personal_best_values[i] = fitness
                    self.personal_best_positions[i] = self.positions[i].copy()
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = self.positions[i].copy()

                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                self.velocities[i] = self.w * self.velocities[i] + self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) + self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = np.clip(self.velocities[i], -self.max_velocity, self.max_velocity)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], -5.0, 5.0)

        return self.global_best_value