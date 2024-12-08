import numpy as np
from scipy.optimize import differential_evolution

class DPHE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.refine_prob = 0.45

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            # Refine the solution with probability 0.45
            if np.random.rand() < self.refine_prob:
                new_bounds = [(self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.uniform(), self.upper_bound) for _ in range(self.dim)]
                new_res = differential_evolution(func, new_bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=10, tol=1e-6)
                if new_res.success:
                    return new_res.x
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