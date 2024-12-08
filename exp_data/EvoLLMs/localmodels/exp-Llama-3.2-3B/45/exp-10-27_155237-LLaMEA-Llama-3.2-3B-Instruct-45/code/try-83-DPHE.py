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

# Probability-based mutation
class DPHE_P:
    def __init__(self, budget, dim, prob):
        self.budget = budget
        self.dim = dim
        self.prob = prob
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            # Select 20% of the population to refine the strategy
            refine_idx = np.random.choice(self.dim, size=int(self.dim * self.prob), replace=False)
            new_x = res.x.copy()
            for i in refine_idx:
                new_x[i] += np.random.uniform(-1, 1, size=1)
            new_x = np.clip(new_x, self.lower_bound, self.upper_bound)
            return new_x
        else:
            return None

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the DPHE_P algorithm
    dphe_p = DPHE_P(budget=100, dim=10, prob=0.45)

    # Optimize the function
    result = dphe_p(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")