import numpy as np

class AdaptiveIMPHS:
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

        def explore_phase(population, mutation_rates):
            new_population = population + np.random.normal(0, mutation_rates, population.shape)
            return np.clip(new_population, self.lower_bound, self.upper_bound)

        def exploit_phase(population, mutation_rates):
            best_idx = np.argmin(evaluate_population(population))
            best_individual = population[best_idx]
            new_population = population + np.random.uniform(-mutation_rates, mutation_rates, population.shape)
            new_population[best_idx] = best_individual
            new_population = np.clip(new_population, self.lower_bound, self.upper_bound)
            return new_population

        population = initialize_population()
        mutation_rates = np.full(self.dim, 0.1)
        for _ in range(self.budget // 2):
            population = exploit_phase(explore_phase(population, mutation_rates), mutation_rates)
            best_idx = np.argmin(evaluate_population(population))
            best_solution = population[best_idx]
            mutation_rates = np.clip(mutation_rates * 0.95, 0.01, 0.2)
        return best_solution