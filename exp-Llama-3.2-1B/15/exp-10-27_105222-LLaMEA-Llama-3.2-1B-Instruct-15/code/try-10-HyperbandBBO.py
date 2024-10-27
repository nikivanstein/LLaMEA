import numpy as np
from scipy.optimize import minimize

class HyperbandBBO:
    def __init__(self, budget, dim, search_space_dim):
        """
        Initialize the HyperbandBBO algorithm.

        Args:
        - budget (int): The maximum number of function evaluations allowed.
        - dim (int): The dimensionality of the search space.
        - search_space_dim (int): The dimensionality of the search space.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = search_space_dim

    def __call__(self, func, algorithm):
        """
        Evaluate the function at a specified point in the search space using the given algorithm.

        Args:
        - func (function): The black box function to evaluate.
        - algorithm (str): The algorithm to use for evaluation. Currently supported: 'bbo', 'hyperband'.

        Returns:
        - function_value (float): The value of the function at the specified point.
        """
        while self.func_evals < self.budget:
            # Sample a new point in the search space using Gaussian distribution
            x = np.random.uniform(*self.search_space, size=self.search_space_dim)
            # Evaluate the function at the new point using the specified algorithm
            func_value = func(x, algorithm)
            # Store the function value and the new point
            self.func_evals += 1
            self.func_evals_evals = func_value
            # Store the new point in the search space
            self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))
        # Evaluate the function at the final point in the search space using the specified algorithm
        func_value = func(self.search_space, algorithm)
        return func_value

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

def bayesian_optimization(func, algorithm, dim):
    """
    Evaluate the function at a specified point in the search space using Bayesian optimization.

    Args:
    - func (function): The black box function to evaluate.
    - algorithm (str): The algorithm to use for evaluation. Currently supported: 'bayesian'.
    - dim (int): The dimensionality of the search space.

    Returns:
    - function_value (float): The value of the function at the specified point.
    """
    while self.func_evals < self.budget:
        # Sample a new point in the search space using Gaussian distribution
        x = np.random.uniform(*self.search_space, size=self.search_space_dim)
        # Evaluate the function at the new point using the specified algorithm
        func_value = func(x, algorithm)
        # Store the function value and the new point
        self.func_evals += 1
        self.func_evals_evals = func_value
        # Store the new point in the search space
        self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))
    # Evaluate the function at the final point in the search space using the specified algorithm
    func_value = func(self.search_space, algorithm)
    return func_value

# Example usage:
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