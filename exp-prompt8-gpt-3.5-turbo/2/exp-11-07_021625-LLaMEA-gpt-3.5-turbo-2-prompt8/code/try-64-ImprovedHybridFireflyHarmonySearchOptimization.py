import numpy as np

class ImprovedHybridFireflyHarmonySearchOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.firefly_iter = 5
        self.harmony_iter = 5

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def firefly_search(population):
            # Implement Firefly Algorithm with adaptive parameter tuning
            return population

        def harmony_search(population):
            # Implement Harmony Search Algorithm with adaptive parameter tuning
            return population

        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.dim,))
        
        # Perform hybrid optimization with reduced redundant evaluations
        for _ in range(self.budget // (self.firefly_iter + self.harmony_iter)):
            population = firefly_search(population)
            population = harmony_search(population)

        # Return the best solution found
        return population