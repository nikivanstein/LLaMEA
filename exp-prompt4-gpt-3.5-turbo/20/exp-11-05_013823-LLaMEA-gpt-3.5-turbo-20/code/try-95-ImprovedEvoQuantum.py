import numpy as np

class ImprovedEvoQuantum:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.elite_ratio = 0.1
        self.mutation_rate = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            fitness_values = np.array([func(individual) for individual in self.population])
            sorted_indices = np.argsort(fitness_values)
            elite_count = int(self.budget * self.elite_ratio)
            elite = self.population[sorted_indices[:elite_count]]

            diversity = np.mean(np.std(elite, axis=0))
            self.mutation_rate = 0.1 / (1 + diversity)

            new_population = np.tile(elite, (int(self.budget / elite_count), 1))

            mutation_mask = np.random.choice([0, 1], size=(self.budget, self.dim), p=[1 - self.mutation_rate, self.mutation_rate])
            new_population += mutation_mask * np.random.normal(0, 1, (self.budget, self.dim))

            self.population = new_population

        best_solution = elite[0]
        return func(best_solution)