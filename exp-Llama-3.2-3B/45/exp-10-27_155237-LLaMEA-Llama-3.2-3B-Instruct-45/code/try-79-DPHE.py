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
            # Select 45% of the best individuals and change their lines
            best_individuals = np.array(res.x[np.argsort(-np.sum(res.fun, axis=0))])
            changed_individuals = np.random.choice(best_individuals, size=int(self.budget * 0.45), replace=False)

            # Change the lines of the selected individuals
            changed_lines = np.random.uniform(self.lower_bound, self.upper_bound, size=(len(changed_individuals), self.dim))
            changed_individuals = changed_individuals[:, np.newaxis] + changed_lines

            # Replace the original individuals with the changed ones
            res.x = np.concatenate((best_individuals, changed_individuals), axis=0)

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