import numpy as np
from scipy.optimize import differential_evolution
import os

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

def refine_strategy(individual, logger):
    # Refine the strategy by changing individual lines with a probability of 0.45
    if np.random.rand() < 0.45:
        for i in range(len(individual)):
            if np.random.rand() < 0.5:
                individual[i] += np.random.uniform(-0.1, 0.1)
            if individual[i] < self.lower_bound:
                individual[i] = self.lower_bound
            elif individual[i] > self.upper_bound:
                individual[i] = self.upper_bound
    return individual

def evaluate_fitness(individual, logger):
    # Evaluate the fitness of the individual
    new_individual = individual
    new_individual = refine_strategy(new_individual, logger)
    return new_individual

def main():
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

    # Evaluate the fitness of the result
    result = evaluate_fitness(result, None)

    # Save the result
    np.save(f"currentexp/aucs-DPHE-0.npy", result)

if __name__ == "__main__":
    main()