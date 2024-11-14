import numpy as np

class DynamicAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, budget // 3)
        self.inertia_weight = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_fitness = np.full(self.population_size, float('inf'))
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.evaluations = 0

    def __call__(self, func):
        self.personal_best_fitness = np.apply_along_axis(func, 1, self.population)
        self.evaluations += self.population_size

        best_index = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_index] < self.global_best_fitness:
            self.global_best_fitness = self.personal_best_fitness[best_index]
            self.global_best_position = self.population[best_index]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                self.inertia_weight = 0.5 + 0.5 * (1 - self.evaluations / self.budget)
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.population[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] + 
                                      cognitive_component + social_component)
                self.population[i] = np.clip(self.population[i] + self.velocities[i], 
                                             self.lower_bound, self.upper_bound)

                current_fitness = func(self.population[i])
                self.evaluations += 1

                if current_fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = current_fitness
                    self.personal_best_positions[i] = self.population[i]

                if current_fitness < self.global_best_fitness:
                    self.global_best_fitness = current_fitness
                    self.global_best_position = self.population[i]

        return self.global_best_position, self.global_best_fitness