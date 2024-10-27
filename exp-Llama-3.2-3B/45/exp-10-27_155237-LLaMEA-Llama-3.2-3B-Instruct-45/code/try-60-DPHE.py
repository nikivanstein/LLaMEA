import numpy as np
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

    # Probabilistic mutation to refine the strategy
    def mutate(individual):
        if np.random.rand() < 0.45:
            # Perturb the individual randomly
            mutated_individual = individual + np.random.uniform(-1, 1, size=individual.shape)
            mutated_individual = np.clip(mutated_individual, self.lower_bound, self.upper_bound)
            return mutated_individual
        else:
            # Copy the individual as is
            return individual

    # Refine the strategy
    refined_individual = mutate(result)
    print("Refined individual:", refined_individual)