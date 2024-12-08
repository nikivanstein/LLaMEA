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

    # Define a probabilistic mutation function
    def probabilistic_mutation(individual, mutation_rate=0.45):
        new_individual = individual.copy()
        for i in range(len(individual)):
            if np.random.rand() < mutation_rate:
                new_individual[i] += np.random.uniform(-1, 1)
                new_individual[i] = max(new_individual[i], self.lower_bound)
                new_individual[i] = min(new_individual[i], self.upper_bound)
        return new_individual

    # Apply probabilistic mutation to the result
    mutated_result = probabilistic_mutation(result, mutation_rate=0.45)
    print("Mutated result:", mutated_result)