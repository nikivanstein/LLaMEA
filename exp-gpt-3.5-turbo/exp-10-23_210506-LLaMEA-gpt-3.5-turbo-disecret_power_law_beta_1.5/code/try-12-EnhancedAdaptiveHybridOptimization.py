import numpy as np

class EnhancedAdaptiveHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso(population):
            # PSO implementation
            pass

        def de(population):
            # DE implementation
            pass

        def update_population(population, new_population):
            # Update population based on fitness
            combined_population = np.concatenate((population, new_population), axis=0)
            combined_population_fitness = np.array([func(individual) for individual in combined_population])
            sorted_indices = np.argsort(combined_population_fitness)
            new_population = combined_population[sorted_indices[:len(population)]]
            return new_population

        # Initialize population using PSO
        population = np.random.uniform(-5.0, 5.0, (self.dim, self.dim))
        population = pso(population)

        for _ in range(self.budget):
            # Perform DE on a subset of the population
            new_population = de(population)

            # Update population based on fitness
            population = update_population(population, new_population)

        # Return best solution found
        return population[np.argmin([func(individual) for individual in population])]