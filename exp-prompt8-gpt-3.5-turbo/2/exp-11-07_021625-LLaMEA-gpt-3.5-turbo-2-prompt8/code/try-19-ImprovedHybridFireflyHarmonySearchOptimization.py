import numpy as np

class ImprovedHybridFireflyHarmonySearchOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def firefly_search(population, max_iter):
            # Implementation of Firefly Algorithm
            pass

        def harmony_search(population, max_iter):
            # Improved Harmony Search Algorithm with adaptive parameter tuning
            harmony_bandwidth = 0.5
            harmony_memory_rate = 0.95
            
            for _ in range(max_iter):
                new_harmony = np.random.uniform(-5.0, 5.0, (self.dim,))
                if objective_function(new_harmony) < objective_function(population):
                    population = new_harmony
                    harmony_bandwidth *= harmony_memory_rate
                
            return population

        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.dim,))
        
        # Perform hybrid optimization
        for _ in range(self.budget):
            population = firefly_search(population, 10)
            population = harmony_search(population, 10)

        # Return the best solution found
        return population