import numpy as np

class EnhancedIMPHS:
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

        def exploit_phase(population, mutation_rates, num_iterations=5):
            for _ in range(num_iterations):
                best_idx = np.argmin(evaluate_population(population))
                best_individual = population[best_idx]
                new_population = population + np.random.uniform(-mutation_rates, mutation_rates, population.shape)
                new_population[best_idx] = best_individual
                new_population = np.clip(new_population, self.lower_bound, self.upper_bound)
                population = new_population
            return population

        population = initialize_population()
        mutation_rates = np.full(self.dim, 0.1)
        for _ in range(self.budget // 2):
            mutation_rates = np.clip(mutation_rates * 0.95 + 0.05 * np.std(evaluate_population(population)), 0.01, 0.2)
            population = exploit_phase(explore_phase(population, mutation_rates), mutation_rates)
        best_idx = np.argmin(evaluate_population(population))
        best_solution = population[best_idx]
        return best_solution