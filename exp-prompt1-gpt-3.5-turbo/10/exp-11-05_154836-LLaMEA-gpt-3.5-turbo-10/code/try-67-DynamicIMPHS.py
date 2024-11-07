# import numpy as np

class DynamicIMPHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.alpha = 0.1

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

        mutation_rates = np.full(self.dim, 0.1)
        population = initialize_population()
        for _ in range(self.budget // 2):
            population = exploit_phase(explore_phase(population, mutation_rates), mutation_rates)
            fitness_values = evaluate_population(population)
            max_fitness_idx = np.argmax(fitness_values)
            min_fitness_idx = np.argmin(fitness_values)
            mutation_rates = mutation_rates + self.alpha * (population[max_fitness_idx] - population[min_fitness_idx])

        best_idx = np.argmin(evaluate_population(population))
        best_solution = population[best_idx]
        return best_solution