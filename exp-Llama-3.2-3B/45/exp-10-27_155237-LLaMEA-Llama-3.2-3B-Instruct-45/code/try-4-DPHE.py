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

# Probabilistic mutation for DPHE algorithm
class DPHE_P:
    def __init__(self, budget, dim, mutation_prob):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_prob = mutation_prob

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            # Select 45% of the population to refine their strategy
            refined_population = np.random.choice(res.x, size=int(0.45 * self.budget), replace=False)
            # Refine the selected population using probabilistic mutation
            refined_population = np.array([self.mutate(individual) for individual in refined_population])
            # Replace the original population with the refined one
            res.x = refined_population
            return res.x
        else:
            return None

    def mutate(self, individual):
        # Select 45% of the dimension to mutate
        indices_to_mutate = np.random.choice(self.dim, size=int(0.45 * self.dim), replace=False)
        mutated_individual = individual.copy()
        for index in indices_to_mutate:
            mutated_individual[index] = np.random.uniform(self.lower_bound, self.upper_bound)
        return mutated_individual

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the DPHE_P algorithm
    dphe_P = DPHE_P(budget=100, dim=10, mutation_prob=0.45)

    # Optimize the function
    result = dphe_P(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")