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

# Modified DPHE class with probabilistic mutation
class DPHEProb:
    def __init__(self, budget, dim, mutation_prob=0.45):
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
            # Apply probabilistic mutation
            if np.random.rand() < self.mutation_prob:
                # Generate a new individual by perturbing the current solution
                new_individual = self.perturb_solution(res.x)
                # Evaluate the new individual
                new_result = self.evaluate_new_individual(func, new_individual)
                # Replace the current solution with the new one if it's better
                if new_result < res.fun:
                    return new_individual
            return res.x
        else:
            return None

    def perturb_solution(self, solution):
        # Generate a new individual by perturbing the current solution
        new_individual = solution.copy()
        for i in range(self.dim):
            # Perturb each dimension with a probability of 0.45
            if np.random.rand() < 0.45:
                new_individual[i] += np.random.uniform(-1, 1)
        return new_individual

    def evaluate_new_individual(self, func, new_individual):
        # Evaluate the new individual
        return -func(new_individual)

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the DPHE algorithm with probabilistic mutation
    dphe_prob = DPHEProb(budget=100, dim=10, mutation_prob=0.45)

    # Optimize the function
    result = dphe_prob(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")