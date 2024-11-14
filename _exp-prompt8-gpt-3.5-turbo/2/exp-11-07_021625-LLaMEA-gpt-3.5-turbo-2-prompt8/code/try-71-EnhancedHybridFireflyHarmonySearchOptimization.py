import numpy as np

class EnhancedHybridFireflyHarmonySearchOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def firefly_search(population, max_iter):
            # Optimized Firefly Algorithm implementation
            pass

        def harmony_search(population, max_iter):
            # Optimized Harmony Search Algorithm implementation
            pass

        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.dim,))
        
        firefly_budget = harmony_budget = self.budget // 2  # Divide budget equally
        
        # Perform hybrid optimization
        for _ in range(firefly_budget):
            population = firefly_search(population, 10)
        
        for _ in range(harmony_budget):
            population = harmony_search(population, 10)

        # Return the best solution found
        return population