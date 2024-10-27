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

# Bayesian Optimization Hyperband Algorithm
class BayesianOptHyperband(HyperbandBBO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        # Initialize the search space with the minimum and maximum values
        x_min, x_max = self.search_space
        # Initialize the best individual and its fitness
        best_individual = np.array([x_min])
        best_fitness = self.func_evals_evals
        # Initialize the best parameter values
        best_params = np.zeros(self.dim)
        # Initialize the bayesian optimization tree
        self.bayes_tree = None

        # Initialize the number of iterations
        self.iterations = 0

        # Define the hyperparameter tuning space
        self.hyperparameter_space = np.linspace(x_min - 0.1, x_max + 0.1, 100)

        # Define the Bayesian optimization algorithm
        self.bayes_opt_algorithm = self._bayes_opt_algorithm

        # Run the Bayesian optimization algorithm
        while self.iterations < self.budget:
            # Evaluate the fitness of the current individual
            fitness = self.func_evals_evals

            # Evaluate the fitness of the current individual at each hyperparameter value
            for i in range(len(self.hyperparameter_space)):
                # Sample a new point in the search space using Gaussian distribution
                x = np.random.uniform(*self.search_space, size=self.search_space_dim)
                # Evaluate the function at the new point
                func_value = func(x)
                # Store the function value and the new point
                self.func_evals_evals += 1
                self.func_evals_evals_evals = func_value

                # Store the new point in the search space
                self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))

            # Evaluate the fitness of the current individual at the hyperparameter values
            for i in range(len(self.hyperparameter_space)):
                # Sample a new point in the search space using Gaussian distribution
                x = np.random.uniform(*self.search_space, size=self.search_space_dim)
                # Evaluate the function at the new point
                func_value = func(x)
                # Store the function value and the new point
                self.func_evals_evals += 1
                self.func_evals_evals_evals = func_value

            # Evaluate the fitness of the current individual at the hyperparameter values
            for i in range(len(self.hyperparameter_space)):
                # Sample a new point in the search space using Gaussian distribution
                x = np.random.uniform(*self.search_space, size=self.search_space_dim)
                # Evaluate the function at the new point
                func_value = func(x)
                # Store the function value and the new point
                self.func_evals_evals += 1
                self.func_evals_evals_evals = func_value

            # Store the current individual and its fitness
            self.best_individual = np.array([x_min])
            self.best_fitness = fitness
            # Store the current best parameter values
            self.best_params = np.copy(best_params)
            # Store the current bayesian optimization tree
            self.bayes_tree = self.bayes_tree

            # Update the best individual and its fitness
            self.best_individual = np.array([x_min])
            self.best_fitness = fitness
            # Update the best parameter values
            self.best_params = np.copy(best_params)
            # Update the bayesian optimization tree
            self.bayes_tree = self._bayes_opt_algorithm(x_min, x_max, self.hyperparameter_space, self.func_evals_evals_evals, best_fitness)

            # Update the number of iterations
            self.iterations += 1

        # Evaluate the fitness of the final individual
        fitness = self.func_evals_evals
        # Return the final individual and its fitness
        return self.best_individual, fitness

    def _bayes_opt_algorithm(self, x_min, x_max, hyperparameter_space, func_evals_evals, best_fitness):
        # Define the Bayesian optimization algorithm
        self.bayes_opt_algorithm = self._bayes_opt_algorithm

        # Initialize the bayesian optimization tree
        self.bayes_tree = self._bayes_opt_algorithm(x_min, x_max, hyperparameter_space, func_evals_evals, best_fitness)

        # Run the Bayesian optimization algorithm
        while True:
            # Evaluate the fitness of the current individual
            fitness = self.func_evals_evals
            # Evaluate the fitness of the current individual at each hyperparameter value
            for i in range(len(hyperparameter_space)):
                # Sample a new point in the search space using Gaussian distribution
                x = np.random.uniform(*self.search_space, size=self.search_space_dim)
                # Evaluate the function at the new point
                func_value = func(x)
                # Store the function value and the new point
                self.func_evals_evals += 1
                self.func_evals_evals_evals = func_value

                # Store the new point in the search space
                self.search_space = (min(self.search_space[0], x), max(self.search_space[1], x))

            # Evaluate the fitness of the current individual at the hyperparameter values
            for i in range(len(hyperparameter_space)):
                # Sample a new point in the search space using Gaussian distribution
                x = np.random.uniform(*self.search_space, size=self.search_space_dim)
                # Evaluate the function at the new point
                func_value = func(x)
                # Store the function value and the new point
                self.func_evals_evals += 1
                self.func_evals_evals_evals = func_value

            # Evaluate the fitness of the current individual at the hyperparameter values
            for i in range(len(hyperparameter_space)):
                # Sample a new point in the search space using Gaussian distribution
                x = np.random.uniform(*self.search_space, size=self.search_space_dim)
                # Evaluate the function at the new point
                func_value = func(x)
                # Store the function value and the new point
                self.func_evals_evals += 1
                self.func_evals_evals_evals = func_value

            # Store the current individual and its fitness
            self.best_individual = np.array([x_min])
            self.best_fitness = fitness
            # Store the current best parameter values
            self.best_params = np.copy(best_params)
            # Update the bayesian optimization tree
            self.bayes_tree = self._bayes_opt_algorithm(x_min, x_max, hyperparameter_space, func_evals_evals_evals, best_fitness)

            # Update the number of iterations
            self.iterations += 1

        # Return the final individual and its fitness
        return self.best_individual, fitness

# Example usage:
def test_func1(x):
    return np.sin(x)

def test_func2(x):
    return x**2 + 2*x + 1

bayesian_opt_hyperband = BayesianOptHyperband(budget=100, dim=10)
optimized_func1 = bayesian_opt_hyperband(test_func1)
optimized_func2 = bayesian_opt_hyperband(test_func2)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot([optimized_func1, optimized_func2], label=['Test Function 1', 'Test Function 2'])
plt.xlabel('Optimized Function Value')
plt.ylabel('Dimensionality')
plt.title('Bayesian Optimization Hyperband Algorithm')
plt.legend()
plt.show()

# Description: Efficient Black Box Optimization using Hyperband and Bayesian Optimization
# Code: 