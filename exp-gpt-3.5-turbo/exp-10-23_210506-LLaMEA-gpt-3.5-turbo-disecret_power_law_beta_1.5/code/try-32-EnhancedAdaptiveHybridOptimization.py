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

        # Adaptive parameter control strategies
        # Initialize population using PSO
        population = pso()

        for _ in range(self.budget):
            # Perform DE on a subset of the population
            new_population = de(population)

            # Update population based on fitness
            population = new_population if func(new_population) < func(population) else population

        # Return best solution found
        return population