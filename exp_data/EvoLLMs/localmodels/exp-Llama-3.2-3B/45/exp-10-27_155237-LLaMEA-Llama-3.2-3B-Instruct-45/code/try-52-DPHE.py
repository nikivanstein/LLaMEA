import numpy as np
from scipy.optimize import differential_evolution

class DPHE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.perturbation_prob = 0.45

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            # Refine the strategy by probabilistically changing individual lines
            refined_individuals = []
            for i in range(self.dim):
                if np.random.rand() < self.perturbation_prob:
                    new_individual = np.copy(res.x)
                    new_individual[i] += np.random.uniform(-1, 1)
                    refined_individuals.append(new_individual)
                else:
                    refined_individuals.append(res.x)

            # Evaluate the refined individuals
            refined_res = differential_evolution(func, bounds, x0=refined_individuals, maxiter=self.budget, tol=1e-6)

            # Update the best individual
            if refined_res.success:
                return refined_res.x
            else:
                return None
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