import numpy as np
from scipy.optimize import minimize

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def update_strategy(self, individual):
        # Refine the strategy by changing the individual lines of the selected solution
        # to refine its strategy
        if individual < 0:
            individual = np.clip(individual, 0, 5)
        elif individual > 5:
            individual = np.clip(individual, -5, 0)
        return individual

    def fit(self, func, max_iter=1000, tol=1e-3):
        # Initialize the population with random individuals
        population = [self.update_strategy(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(100)]
        for _ in range(max_iter):
            # Evaluate the fitness of each individual
            fitness = [func(individual) for individual in population]
            # Select the fittest individuals
            fittest = [population[i] for i, f in enumerate(fitness) if f == max(fitness)]
            # Create a new population by breeding the fittest individuals
            new_population = [self.update_strategy(individual) for individual in fittest]
            # Replace the old population with the new one
            population = new_population
        return population

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Evaluate the fitness of the selected solution
fitness = func(result)
print(fitness)