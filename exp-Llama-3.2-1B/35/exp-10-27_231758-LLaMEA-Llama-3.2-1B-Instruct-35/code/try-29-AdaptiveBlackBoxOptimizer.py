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

    def adaptive_search(self, func, budget, dim, max_iter=100, tol=1e-6):
        """
        Adaptive Black Box Optimization using Adaptive Search.

        Parameters:
        - func: The black box function to optimize.
        - budget: The number of function evaluations allowed.
        - dim: The dimensionality of the search space.
        - max_iter: The maximum number of iterations. Defaults to 100.
        - tol: The tolerance for convergence. Defaults to 1e-6.

        Returns:
        - A tuple containing the optimized function value and the optimized function.
        """
        # Initialize the population with random values
        population = np.random.rand(self.dim) + np.arange(self.dim)

        # Evolve the population using the adaptive search algorithm
        for _ in range(max_iter):
            # Evaluate the function at the current population
            func(population)

            # Select the fittest individual
            idx = np.argmin(np.abs(population))
            population[idx] = func(population[idx])

            # Check for convergence
            if np.all(population == func(population)):
                break

        # Return the optimized function value and the optimized function
        return func(population), population

    def adaptive_search_with_refinement(self, func, budget, dim, max_iter=100, tol=1e-6):
        """
        Adaptive Black Box Optimization using Adaptive Search with refinement.

        Parameters:
        - func: The black box function to optimize.
        - budget: The number of function evaluations allowed.
        - dim: The dimensionality of the search space.
        - max_iter: The maximum number of iterations. Defaults to 100.
        - tol: The tolerance for convergence. Defaults to 1e-6.

        Returns:
        - A tuple containing the optimized function value and the optimized function.
        """
        # Initialize the population with random values
        population = np.random.rand(self.dim)

        # Evolve the population using the adaptive search algorithm
        for _ in range(max_iter):
            # Evaluate the function at the current population
            func(population)

            # Select the fittest individual
            idx = np.argmin(np.abs(population))
            population[idx] = func(population[idx])

            # Check for convergence
            if np.all(population == func(population)):
                break

            # Refine the search space using the adaptive search algorithm
            refined_population = self.adaptive_search(func, budget, dim, max_iter=100, tol=1e-6)
            population = refined_population

        # Return the optimized function value and the optimized function
        return func(population), population