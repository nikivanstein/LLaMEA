import numpy as np
from scipy.optimize import differential_evolution
import random

class DPHE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            return res.x
        else:
            return None

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

    # Refine the strategy with probability 0.45
    if random.random() < 0.45:
        # Select a random individual from the current population
        selected_individual = random.choice([i for i in result if i is not None])

        # If the individual is not None, refine its strategy
        if selected_individual is not None:
            # Calculate the probability of changing each dimension
            prob_change = [random.random() < 0.45 for _ in range(self.dim)]

            # Change each dimension with the calculated probability
            new_individual = selected_individual.copy()
            for i in range(self.dim):
                if prob_change[i]:
                    new_individual[i] = np.random.uniform(self.lower_bound, self.upper_bound)
            return new_individual
    return None