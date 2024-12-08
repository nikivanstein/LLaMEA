import numpy as np

class AdaptiveParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(10, 10 * dim // 3)
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.inertia = 0.7
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best = np.copy(self.population)
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best = None
        self.global_best_fitness = np.inf

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            fitness_values = np.apply_along_axis(func, 1, self.population)
            evaluations += self.population_size

            for i in range(self.population_size):
                if fitness_values[i] < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness_values[i]
                    self.personal_best[i] = self.population[i]

                if fitness_values[i] < self.global_best_fitness:
                    self.global_best_fitness = fitness_values[i]
                    self.global_best = self.population[i]

            inertia_weight = np.linspace(0.9, 0.4, self.budget // self.population_size)
            self.inertia = inertia_weight[min(evaluations // self.population_size, len(inertia_weight) - 1)]

            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_component = self.c1 * r1 * (self.personal_best[i] - self.population[i])
                social_component = self.c2 * r2 * (self.global_best - self.population[i])
                self.velocities[i] = self.inertia * self.velocities[i] + cognitive_component + social_component
                self.population[i] = np.clip(self.population[i] + self.velocities[i], *self.bounds)

        return self.global_best, self.global_best_fitness