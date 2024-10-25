import numpy as np

class StochasticPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 8 * dim
        self.bounds = (-5.0, 5.0)
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w = 0.7  # inertia weight
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_fitness = np.full(self.pop_size, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_fitness = np.inf

    def __call__(self, func):
        evaluations = 0

        # Initialize fitness values for the initial population
        for i in range(self.pop_size):
            if evaluations >= self.budget:
                break
            fitness_value = func(self.population[i])
            evaluations += 1
            if fitness_value < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = fitness_value
                self.personal_best_positions[i] = self.population[i]
                if fitness_value < self.global_best_fitness:
                    self.global_best_fitness = fitness_value
                    self.global_best_position = self.population[i]

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                # Update velocity using adaptive control
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.population[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component

                # Introduce stochastic perturbation
                perturbation = np.random.normal(0, 0.1, self.dim)
                self.population[i] += self.velocities[i] + perturbation
                self.population[i] = np.clip(self.population[i], self.bounds[0], self.bounds[1])

                # Evaluate new positions
                fitness_value = func(self.population[i])
                evaluations += 1

                # Update personal and global bests
                if fitness_value < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness_value
                    self.personal_best_positions[i] = self.population[i]
                    if fitness_value < self.global_best_fitness:
                        self.global_best_fitness = fitness_value
                        self.global_best_position = self.population[i]

        return self.global_best_position, self.global_best_fitness