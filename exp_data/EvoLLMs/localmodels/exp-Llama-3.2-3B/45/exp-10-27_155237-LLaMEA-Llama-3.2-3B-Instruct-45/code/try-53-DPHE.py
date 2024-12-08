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
            # Refine the solution using probability-based line search
            probabilities = np.random.uniform(0.0, 1.0, size=self.dim)
            refined_x = res.x
            for i in range(self.dim):
                if probabilities[i] > 0.45:
                    # Perturb the solution using a small random step
                    perturbation = np.random.uniform(-0.1, 0.1, size=self.dim)
                    refined_x[i] += perturbation
                    # Ensure the solution stays within the bounds
                    refined_x[i] = np.clip(refined_x[i], self.lower_bound, self.upper_bound)
            return refined_x
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