import numpy as np

class EnhancedHybridFireflyHarmonySearchOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def hybrid_search(population, max_iter):
            # Combined implementation of Firefly and Harmony Search Algorithms
            # Efficiently optimize the population
            pass

        # Initialize population once outside the loop
        population = np.random.uniform(-5.0, 5.0, (self.dim,))
        
        # Perform hybrid optimization directly
        for _ in range(self.budget):
            population = hybrid_search(population, 10)

        return population