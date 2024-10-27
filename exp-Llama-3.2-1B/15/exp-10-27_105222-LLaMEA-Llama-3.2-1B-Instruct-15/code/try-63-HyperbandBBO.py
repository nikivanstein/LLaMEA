import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Sample a new point in the search space using Gaussian distribution
            x = np.random.uniform(*self.search_space, size=self.search_space_dim)
            # Evaluate the function at the new point
            func_value = func(x)
            # Store the function value and the new point
            self.func_evals += 1
            self.func_evals_evals = func_value
            # Store the new point in the search space
            self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))
        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space)
        return func_value

    def optimize_hyperband(self, func):
        # Define the hyperband search space
        hyperband_search_space = [np.linspace(self.search_space[0], self.search_space[1], 10) for _ in range(10)]

        # Initialize the population with random points in the hyperband search space
        population = [np.random.uniform(hyperband_search_space[i][0], hyperband_search_space[i][1]) for i in range(self.dim)]

        # Define the Bayesian optimization algorithm
        bayesian_optimizer = minimize(lambda x: -func(x), population, method="SLSQP", bounds=[(hyperband_search_space[i][0], hyperband_search_space[i][1]) for i in range(self.dim)])

        # Update the population with the best individual from the Bayesian optimization algorithm
        bayesian_optimizer.x = bayesian_optimizer.x[0]
        bayesian_optimizer.fun = bayesian_optimizer.fun[0]

        # Evaluate the function at the new point in the population
        population_evals = len(population)
        population_evals_evals = -bayesian_optimizer.fun
        population_evals = np.random.uniform(0, population_evals_evals)
        population = population[int(population_evals_evals):]

        # Update the search space for the next iteration
        self.search_space = (min(self.search_space[0], population[0]), max(self.search_space[1], population[0]))

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10)
optimized_func1 = hyperband(test_func1)
optimized_func2 = hyperband(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()