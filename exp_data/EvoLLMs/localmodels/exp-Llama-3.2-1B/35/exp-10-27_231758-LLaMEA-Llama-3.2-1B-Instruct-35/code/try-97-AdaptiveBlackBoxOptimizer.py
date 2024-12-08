import numpy as np
import random
from scipy.optimize import differential_evolution

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

    def adaptive_differential_evolution(self, func, bounds, initial_guess, budget):
        """
        Adaptive Differential Evolution Optimization Algorithm

        Parameters:
        func (function): The objective function to be optimized
        bounds (list): The search space bounds for each dimension
        initial_guess (list): The initial guess for each dimension
        budget (int): The maximum number of function evaluations

        Returns:
        solution (list): The optimized solution
        """
        # Define the bounds and initial guess
        self.bounds = bounds
        self.initial_guess = initial_guess

        # Run the differential evolution algorithm
        result = differential_evolution(func, self.bounds, initial_guess=self.initial_guess, maxiter=budget)
        return result.x

    def adaptive_differential_evolution_refined(self, func, bounds, initial_guess, budget):
        """
        Adaptive Differential Evolution Optimization Algorithm with Refining Strategy

        Parameters:
        func (function): The objective function to be optimized
        bounds (list): The search space bounds for each dimension
        initial_guess (list): The initial guess for each dimension
        budget (int): The maximum number of function evaluations

        Returns:
        solution (list): The optimized solution
        """
        # Define the bounds and initial guess
        self.bounds = bounds
        self.initial_guess = initial_guess

        # Run the differential evolution algorithm
        result = differential_evolution(func, self.bounds, initial_guess=self.initial_guess, maxiter=budget)

        # Refine the solution using a strategy to balance exploration and exploitation
        # (0.35: 35% exploration, 65% exploitation)
        exploration = 0.35
        exploitation = 0.65
        if np.random.rand() < exploration:
            # Explore the search space
            solution = initial_guess
            for _ in range(10):
                idx = np.random.choice(len(self.bounds))
                solution = np.append(solution, func(solution[idx]))
            solution = np.delete(solution, idx)
        else:
            # Exploit the search space
            solution = result.x

        return solution

# Description: Adaptive Black Box Optimization Algorithm using Adaptive Differential Evolution
# Code: 