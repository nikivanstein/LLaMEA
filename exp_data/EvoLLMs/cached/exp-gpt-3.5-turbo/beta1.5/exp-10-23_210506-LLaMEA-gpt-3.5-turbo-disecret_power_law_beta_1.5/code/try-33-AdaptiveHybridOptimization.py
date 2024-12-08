import numpy as np

class AdaptiveHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso(population):
            # PSO implementation
            return updated_population

        def de(population):
            # DE implementation
            return updated_population

        # Adaptive parameter control strategies
        population = np.random.uniform(-5.0, 5.0, (self.dim, self.dim))

        for _ in range(self.budget):
            new_population = de(population)
            population = pso(population)

        return population[np.argmin([func(x) for x in population])]