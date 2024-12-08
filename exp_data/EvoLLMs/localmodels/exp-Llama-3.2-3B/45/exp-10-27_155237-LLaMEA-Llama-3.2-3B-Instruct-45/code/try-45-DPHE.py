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

    def refine(self, result, probability=0.45):
        if result is not None:
            new_result = result.copy()
            for i in range(self.dim):
                if np.random.rand() < probability:
                    new_result[i] = np.random.uniform(self.lower_bound, self.upper_bound)
            return new_result
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

    # Refine the result with probability 0.45
    refined_result = dphe.refine(result, probability=0.45)

    # Print the refined result
    if refined_result is not None:
        print("Refined optimal solution:", refined_result)
    else:
        print("Failed to converge")