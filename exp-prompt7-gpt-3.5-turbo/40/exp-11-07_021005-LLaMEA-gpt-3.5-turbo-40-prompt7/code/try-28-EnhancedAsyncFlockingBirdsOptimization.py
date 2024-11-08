import numpy as np
import concurrent.futures

class EnhancedAsyncFlockingBirdsOptimization:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.population_size, self.max_velocity, self.c1, self.c2, self.w = budget, dim, 20, 0.1, 2.0, 2.0, 0.5
        self.positions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-self.max_velocity, self.max_velocity, (self.population_size, self.dim))
        self.personal_best_positions, self.personal_best_values = self.positions.copy(), np.full(self.population_size, np.inf)
        self.global_best_position, self.global_best_value = np.zeros(self.dim), np.inf

    def evaluate_fitness(self, func, i):
        return func(self.positions[i])

    def __call__(self, func):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_index = {executor.submit(self.evaluate_fitness, func, i): i for i in range(self.population_size)}
            for future in concurrent.futures.as_completed(future_to_index):
                i = future_to_index[future]
                fitness = future.result()
                if fitness < self.personal_best_values[i]:
                    self.personal_best_values[i], self.personal_best_positions[i] = fitness, self.positions[i]
                if fitness < self.global_best_value:
                    self.global_best_value, self.global_best_position = fitness, self.positions[i]

                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocities[i] = self.w * self.velocities[i] + self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i]) + self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = np.clip(self.velocities[i], -self.max_velocity, self.max_velocity)
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], -5.0, 5.0)

        return self.global_best_value