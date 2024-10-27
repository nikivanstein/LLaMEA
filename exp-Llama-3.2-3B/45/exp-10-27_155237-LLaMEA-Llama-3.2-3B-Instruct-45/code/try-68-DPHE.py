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

# Modified DPHE algorithm with probability-based mutation
class ModifiedDPHE:
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
            # Apply mutation with probability
            mutated_individual = self.mutate(res.x, self.mutation_prob)
            if np.random.rand() < self.mutation_prob:
                return mutated_individual
            else:
                return res.x
        else:
            return None

    def mutate(self, individual, mutation_prob):
        mutated_individual = individual.copy()
        for i in range(self.dim):
            if np.random.rand() < mutation_prob:
                mutated_individual[i] += np.random.uniform(-1, 1)
                mutated_individual[i] = max(self.lower_bound, min(self.upper_bound, mutated_individual[i]))
        return mutated_individual

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the modified DPHE algorithm
    modified_dphe = ModifiedDPHE(budget=100, dim=10, mutation_prob=0.45)

    # Optimize the function
    result = modified_dphe(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")