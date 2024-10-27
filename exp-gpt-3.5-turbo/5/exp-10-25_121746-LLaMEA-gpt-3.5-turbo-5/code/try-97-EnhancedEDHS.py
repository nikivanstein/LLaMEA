import numpy as np

class EnhancedEDHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20  # Initial population size
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.fitness_values = np.zeros(self.population_size)
        self.global_best = None
        self.global_best_fitness = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            self.evaluate_population(func)
            self.update_global_best()
            self.adjust_population_size()
            self.update_population()

        return self.global_best

    def evaluate_population(self, func):
        for i in range(self.population_size):
            self.fitness_values[i] = func(self.population[i])

    def update_global_best(self):
        min_idx = np.argmin(self.fitness_values)
        if self.fitness_values[min_idx] < self.global_best_fitness:
            self.global_best_fitness = self.fitness_values[min_idx]
            self.global_best = self.population[min_idx]

    def adjust_population_size(self):
        diversity_threshold = 0.05
        diversity = np.std(self.fitness_values)
        if diversity < diversity_threshold:
            self.population_size += 5
        elif diversity > 0.1:
            self.population_size -= 5

    def update_population(self):
        # Update population based on the optimization strategy
        pass