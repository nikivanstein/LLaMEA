import numpy as np

class EvoQuantum:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.elite_ratio = 0.1  # Initial elite ratio
        
    def __call__(self, func):
        for _ in range(self.budget):
            fitness_values = np.array([func(individual) for individual in self.population])
            sorted_indices = np.argsort(fitness_values)
            diversity = np.mean(np.std(self.population, axis=0))
            self.elite_ratio = 0.1 + 0.4 * (1 - np.exp(-diversity))  # Dynamic adaptation of elite ratio based on diversity
            elite_count = int(self.budget * self.elite_ratio)
            elite = self.population[sorted_indices[:elite_count]]
            new_population = np.tile(elite, (int(self.budget / elite_count), 1))

            mutation_rate = 0.1 / (1 + diversity)
            mutation_mask = np.random.choice([0, 1], size=(self.budget, self.dim), p=[1 - mutation_rate, mutation_rate])
            new_population += mutation_mask * np.random.normal(0, 1, (self.budget, self.dim))

            self.population = new_population
        best_solution = elite[0]
        return func(best_solution)