import numpy as np
from scipy.optimize import differential_evolution

class DPHE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.refinement_probability = 0.45

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            # Refine the solution with probability 0.45
            if np.random.rand() < self.refinement_probability:
                new_individual = self.refine_solution(res.x)
                return new_individual
            else:
                return res.x
        else:
            return None

    def refine_solution(self, solution):
        new_individual = solution.copy()
        for i in range(self.dim):
            if np.random.rand() < self.refinement_probability:
                # Perturb the individual with a small mutation
                mutation = np.random.uniform(-0.1, 0.1, size=self.dim)
                new_individual = new_individual + mutation
                # Ensure the individual stays within the bounds
                new_individual = np.clip(new_individual, self.lower_bound, self.upper_bound)
        return new_individual

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