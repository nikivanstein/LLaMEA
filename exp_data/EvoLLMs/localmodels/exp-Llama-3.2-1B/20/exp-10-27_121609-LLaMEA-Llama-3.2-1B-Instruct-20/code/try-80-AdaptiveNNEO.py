# BBOB Optimization Algorithm: Novel Hybrid Approach
# Description: A novel hybrid algorithm that combines evolutionary and adaptive strategies to optimize black box functions.
# Code: 
# ```python
import numpy as np

class AdaptiveNNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutation(x):
            if np.random.rand() < 0.2:
                return x + np.random.uniform(-5.0, 5.0)
            else:
                return x

        def adaptive_strategy(x):
            if np.random.rand() < 0.2:
                return mutation(x)
            else:
                return x

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = adaptive_strategy(x)

        return self.fitnesses

# Example usage:
def func(x):
    return x**2 + 3*x + 2

algorithm = AdaptiveNNEO(100, 10)
result = algorithm(func)
print(result)