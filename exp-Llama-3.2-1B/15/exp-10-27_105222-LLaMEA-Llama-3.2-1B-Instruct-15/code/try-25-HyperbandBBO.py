import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

class HyperbandBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.func_evals = 0
        self.search_space_dim_init = self.dim
        self.search_space_dim_step = 0.1

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Sample a new point in the search space using Gaussian distribution
            x = np.random.uniform(*self.search_space, size=self.search_space_dim_init)
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

    def refine_strategy(self, func, initial_strategy):
        # Update the search space dimension
        self.search_space_dim_init = initial_strategy.search_space_dim
        # Update the step size for the Gaussian distribution
        self.search_space_dim_step = initial_strategy.search_space_dim_step
        # Refine the search space using the updated parameters
        new_strategy = HyperbandBBO(self.budget, self.search_space_dim_init)
        # Run the search space refinement
        new_strategy_func_evals = new_strategy(func)
        # Refine the strategy based on the fitness values
        new_strategy.search_space_dim = np.log(new_strategy_func_evals / new_strategy_func_evals_evals)
        new_strategy.search_space_dim_step = new_strategy.search_space_dim_step / 2
        return new_strategy

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10)
initial_strategy = HyperbandBBO(budget=100, dim=10)
optimized_func1 = hyperband(test_func1)
optimized_func2 = hyperband(test_func2)
refined_strategy = hyperband.refine_strategy(optimized_func1, initial_strategy)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()

# Plot the refinement process
plt.figure(figsize=(8, 6))
plt.plot([initial_strategy.func_evals_evals, refined_strategy.func_evals_evals], label=['Initial', 'Refined'])
plt.xlabel('Number of Function Evaluations')
plt.ylabel('Function Value')
plt.title('Refinement Process')
plt.legend()
plt.show()