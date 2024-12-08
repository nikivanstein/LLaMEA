import numpy as np

class ENHMPHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

        def evaluate_population(population):
            return np.array([func(individual) for individual in population])

        def evolve_population(population, mutation_rate):
            return np.clip(population + np.random.normal(0, mutation_rate, population.shape), self.lower_bound, self.upper_bound)

        population = initialize_population()
        for _ in range(self.budget // 2):
            mutation_rate = np.random.uniform(0.05, 0.2, self.dim)
            population = evolve_population(population, mutation_rate)
            best_idx = np.argmin(evaluate_population(population))
            best_individual = population[best_idx]
            population = evolve_population(population, np.random.uniform(0, 0.1))
            population[best_idx] = best_individual
        best_idx = np.argmin(evaluate_population(population))
        best_solution = population[best_idx]
        return best_solution