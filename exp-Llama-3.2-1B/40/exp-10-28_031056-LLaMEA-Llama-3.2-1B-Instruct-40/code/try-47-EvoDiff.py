import numpy as np
from scipy.optimize import differential_evolution

class EvoDiff:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.init_population()

    def init_population(self):
        # Initialize the population with random solutions
        return np.random.uniform(-5.0, 5.0, self.dim) + np.random.normal(0, 1, self.dim)

    def __call__(self, func):
        # Evaluate the black box function with the current population
        func_values = np.array([func(x) for x in self.population])

        # Select the fittest solutions
        fittest_indices = np.argsort(func_values)[::-1][:self.population_size]

        # Evolve the population using evolutionary differential evolution
        bounds = [(-5.0, 5.0), (-5.0, 5.0)] * self.dim  # Search space
        res = differential_evolution(self.budget, bounds, args=(func_values,), x0=fittest_indices, maxiter=1000)

        # Refine the strategy by changing the lines of the selected solution
        refined_individual = res.x[0]
        refined_individual = refined_individual[:self.dim] + np.random.normal(0, 1, self.dim) * res.x[1]
        refined_individual = refined_individual + np.random.normal(0, 1, self.dim) * res.x[1]

        # Replace the old population with the new one
        self.population = np.concatenate((self.population, refined_individual), axis=0)
        self.population = np.concatenate((self.population, refined_individual), axis=0)

        # Evaluate the function with the final population
        func_values = np.array([func(x) for x in self.population])
        return func_values

# Example usage:
problem = RealSingleObjectiveProblem(1, 1, func=lambda x: x**2, iid=1, dim=1)
algo = EvoDiff(1000, 1, population_size=100, mutation_rate=0.01)
best_func_values = np.array([func(x) for x in algo.population])
print("Best function values:", best_func_values)