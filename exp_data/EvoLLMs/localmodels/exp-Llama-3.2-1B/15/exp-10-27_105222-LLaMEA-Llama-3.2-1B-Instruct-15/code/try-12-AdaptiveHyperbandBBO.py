import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class AdaptiveHyperbandBBO:
    def __init__(self, budget, dim, initial_strategy, learning_rate, decay_rate):
        """
        Initialize the Adaptive Hyperband Bayesian Optimization algorithm.

        Args:
            budget (int): Maximum number of function evaluations.
            dim (int): Dimensionality of the search space.
            initial_strategy (str): Initial strategy for the algorithm (e.g., 'random', 'grid').
            learning_rate (float): Learning rate for the Bayesian optimization.
            decay_rate (float): Decay rate for the Bayesian optimization.
        """
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.search_space_dim = self.dim
        self.search_space_bounds = self.search_space
        self.search_space_bounds_dim = self.dim
        self.func_evals = 0
        self.search_space = initial_strategy
        self.search_space_bounds = initial_strategy
        self.search_space_bounds_dim = initial_strategy
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.bayes = None
        self.bayes_history = []

    def __call__(self, func):
        """
        Optimize the black box function using the Adaptive Hyperband Bayesian Optimization algorithm.

        Args:
            func (function): Black box function to optimize.

        Returns:
            float: Optimized function value.
        """
        while self.func_evals < self.budget:
            # Sample a new point in the search space using Gaussian distribution
            if self.search_space == 'random':
                x = np.random.uniform(*self.search_space_bounds, size=self.search_space_bounds_dim)
            elif self.search_space == 'grid':
                x = np.random.uniform(self.search_space_bounds[0], self.search_space_bounds[1], size=self.search_space_bounds_dim)
            else:
                x = np.random.uniform(self.search_space_bounds[0], self.search_space_bounds[1], size=self.search_space_bounds_dim)
            # Evaluate the function at the new point
            func_value = func(x)
            # Store the function value and the new point
            self.func_evals += 1
            self.func_evals_evals = func_value
            # Store the new point in the search space
            self.search_space = self.search_space_bounds if self.search_space == 'random' else self.search_space_bounds_dim
            self.search_space_bounds = self.search_space_bounds_dim
            # Store the new point in the search space bounds
            self.search_space_bounds_dim = self.search_space_bounds_dim
            # Update the Bayesian optimization
            self.bayes.update(x, func_value)
            # Update the Bayesian optimization history
            self.bayes_history.append(func_value)
            # Update the search space bounds using Bayesian optimization
            self.update_search_space_bounds()

        # Evaluate the function at the final point in the search space
        func_value = func(self.search_space_bounds)
        return func_value

    def update_search_space_bounds(self):
        """
        Update the search space bounds using Bayesian optimization.
        """
        # Get the current Bayesian optimization history
        bayes_history = self.bayes_history
        # Calculate the mean and standard deviation of the Bayesian optimization history
        mean = np.mean(bayes_history)
        std = np.std(bayes_history)
        # Update the search space bounds using Bayesian optimization
        self.search_space_bounds = (mean - std * self.learning_rate, mean + std * self.learning_rate)
        # Update the search space bounds history
        self.bayes_history = []

    def evaluate_fitness(self, individual):
        """
        Evaluate the fitness of the individual using the Adaptive Hyperband Bayesian Optimization algorithm.

        Args:
            individual (list): Individual to evaluate.

        Returns:
            float: Fitness value of the individual.
        """
        # Evaluate the function at the individual
        func_value = self.func(individual)
        # Store the function value and the individual
        self.func_evals_evals = func_value
        # Store the individual in the search space
        self.search_space = self.search_space_bounds if self.search_space == 'random' else self.search_space_bounds_dim
        self.search_space_bounds = self.search_space_bounds_dim
        # Store the individual in the search space bounds
        self.search_space_bounds_dim = self.search_space_bounds_dim
        # Update the Bayesian optimization
        self.bayes.update(individual, func_value)
        # Update the Bayesian optimization history
        self.bayes_history.append(func_value)
        # Return the fitness value of the individual
        return func_value

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

adaptive_hyperband = AdaptiveHyperbandBBO(budget=100, dim=10, initial_strategy='random', learning_rate=0.1, decay_rate=0.1)
optimized_func1 = adaptive_hyperband(test_func1)
optimized_func2 = adaptive_hyperband(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Black Box Optimization using Adaptive Hyperband and Bayesian Optimization')
plt.legend()
plt.show()