import numpy as np

class AQ_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_fitness = np.full(self.pop_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient

    def evaluate(self, func, solution):
        self.evaluations += 1
        return func(solution)

    def update_velocity(self, idx):
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[idx] - self.population[idx])
        social_velocity = self.c2 * r2 * (self.global_best_position - self.population[idx])
        self.velocities[idx] = self.w * self.velocities[idx] + cognitive_velocity + social_velocity

    def quantum_update(self, idx):
        phi = np.random.uniform(-1, 1, self.dim)
        l = np.linalg.norm(self.global_best_position - self.population[idx]) / 2
        quantum_step = np.random.normal(0, l, self.dim)
        return self.global_best_position + phi * quantum_step

    def __call__(self, func):
        np.random.seed(42)

        # Initial evaluation
        for i in range(self.pop_size):
            current_fitness = self.evaluate(func, self.population[i])
            self.personal_best_fitness[i] = current_fitness
            self.personal_best_positions[i] = self.population[i]
            if current_fitness < self.global_best_fitness:
                self.global_best_fitness = current_fitness
                self.global_best_position = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if np.random.rand() < 0.5:
                    self.update_velocity(i)
                    self.population[i] += self.velocities[i]
                else:
                    self.population[i] = self.quantum_update(i)

                # Ensure particles are within bounds
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

                # Evaluate the new position
                current_fitness = self.evaluate(func, self.population[i])

                # Update personal best
                if current_fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = current_fitness
                    self.personal_best_positions[i] = self.population[i]

                # Update global best
                if current_fitness < self.global_best_fitness:
                    self.global_best_fitness = current_fitness
                    self.global_best_position = self.population[i]

        return self.global_best_position