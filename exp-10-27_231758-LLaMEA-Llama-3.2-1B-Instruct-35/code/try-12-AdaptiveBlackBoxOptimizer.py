import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

    def adaptive_search(self, func, budget, dim):
        """
        Adaptive Black Box Optimization using Adaptive Search Algorithm.

        The algorithm uses a combination of greedy search and adaptive search to improve the convergence rate.

        Parameters:
        func (function): The black box function to optimize.
        budget (int): The maximum number of function evaluations allowed.
        dim (int): The dimensionality of the search space.

        Returns:
        tuple: A tuple containing the optimized function values and the number of function evaluations.
        """
        # Initialize the population with random function values
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, dim))
        for i in range(self.budget):
            population[i] = func(population[i])

        # Initialize the best function value and its index
        best_func_value = np.min(np.abs(population))
        best_func_idx = np.argmin(np.abs(population))

        # Perform adaptive search to refine the search space
        for _ in range(10):
            # Select the next function value based on the adaptive search strategy
            idx = np.argmin(np.abs(population))
            next_func_value = func(population[idx])

            # Update the population with the new function value
            population = np.roll(population, 1, axis=0)
            population[0] = next_func_value

            # Update the best function value and its index
            best_func_value = np.min(np.abs(population))
            best_func_idx = np.argmin(np.abs(population))

        return best_func_value, population

# One-line description with the main idea:
# AdaptiveBlackBoxOptimizer: Adaptive Search Algorithm for Black Box Optimization
# Code: 