import numpy as np
import random
from scipy.optimize import differential_evolution

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

    # Define the refinement probability
    refinement_prob = 0.45

    # Refine the solution
    if random.random() < refinement_prob:
        # Generate a new individual
        new_individual = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)

        # Evaluate the fitness of the new individual
        new_fitness = func(new_individual)

        # Evaluate the fitness of the current individual
        current_fitness = func(result)

        # Calculate the probability of accepting the new individual
        prob = np.exp((current_fitness - new_fitness) / (self.budget * (self.upper_bound - self.lower_bound)))

        # Accept the new individual with the calculated probability
        if random.random() < prob:
            result = new_individual

    # Print the refined result
    print("Refined optimal solution:", result)