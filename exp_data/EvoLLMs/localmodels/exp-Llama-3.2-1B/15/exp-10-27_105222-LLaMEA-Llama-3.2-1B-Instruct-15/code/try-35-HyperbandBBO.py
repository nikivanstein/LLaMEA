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
        self.search_space_dim = self.dim

    def __call__(self, func, algorithm):
        if algorithm == 'BayesianOptimization':
            # Bayesian Optimization algorithm
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
        elif algorithm == 'Hyperband':
            # Hyperband algorithm
            while self.func_evals < self.budget:
                # Calculate the probability of each point in the search space
                probabilities = np.exp(-((self.search_space[0] - x) / 1.5)**2) * np.exp(((x - self.search_space[1]) / 1.5)**2)
                # Select the point with the highest probability
                selected_point = np.random.choice(len(self.search_space), p=probabilities)
                # Evaluate the function at the selected point
                func_value = func(selected_point)
                # Store the function value and the selected point
                self.func_evals += 1
                self.func_evals_evals = func_value
                # Store the selected point in the search space
                self.search_space[selected_point] = (self.search_space[selected_point][0], self.search_space[selected_point][1])
        else:
            raise ValueError("Invalid algorithm. Supported algorithms are 'BayesianOptimization' and 'Hyperband'.")

    def get_best_point(self, func, budget):
        # Get the best point in the search space
        best_point = np.argmax(func(self.search_space))
        # Evaluate the function at the best point
        best_func_value = func(self.search_space[best_point])
        return best_point, best_func_value

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

hyperband = HyperbandBBO(budget=100, dim=10)
optimized_func1 = hyperband(test_func1, 'BayesianOptimization')
optimized_func2 = hyperband(test_func2, 'Hyperband')

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Hyperband and Bayesian Optimization')
plt.legend()
plt.show()