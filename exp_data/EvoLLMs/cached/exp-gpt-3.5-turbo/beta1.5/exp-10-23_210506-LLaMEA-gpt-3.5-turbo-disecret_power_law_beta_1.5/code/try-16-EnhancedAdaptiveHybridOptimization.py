import numpy as np

class EnhancedAdaptiveHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso():
            # Enhanced PSO implementation
            pass

        def de():
            # Enhanced DE implementation
            pass

        # Adaptive parameter control strategies
        # Initialize population using Enhanced PSO
        population = pso()

        for _ in range(self.budget):
            # Perform Enhanced DE on a subset of the population
            new_population = de()

            # Update population based on fitness
            population = update_population(population, new_population)

        # Return best solution found
        return best_solution(population)