import numpy as np
from scipy.optimize import differential_evolution
import random

class DPHE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.refine_prob = 0.45

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            # Refine the solution with a probability of 0.45
            if random.random() < self.refine_prob:
                new_individual = self.refine(res.x)
                return new_individual
            else:
                return res.x
        else:
            return None

    def refine(self, x):
        new_individual = []
        for i in range(self.dim):
            if random.random() < self.refine_prob:
                # Perturb the current value by a small amount
                new_individual.append(x[i] + np.random.uniform(-0.1, 0.1))
            else:
                new_individual.append(x[i])
        return new_individual

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the DPHE algorithm
    dphe = DPHE(budget=100, dim=10)

    # Optimize the function
    result = dphe(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")