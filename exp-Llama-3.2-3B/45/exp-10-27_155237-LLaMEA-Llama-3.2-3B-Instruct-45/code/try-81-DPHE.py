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

# Modified DPHE class to include probabilistic mutation
class ProbabilisticDPHE(DPHE):
    def __init__(self, budget, dim, mutation_prob):
        super().__init__(budget, dim)
        self.mutation_prob = mutation_prob

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            # Apply probabilistic mutation to the solution
            if np.random.rand() < self.mutation_prob:
                # Generate a new solution by perturbing the current solution
                new_solution = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
                new_solution += np.random.uniform(-1, 1, size=self.dim) * np.abs(np.random.uniform(-1, 1, size=self.dim))
                new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
                # Evaluate the new solution
                new_res = differential_evolution(func, bounds, x0=new_solution, maxiter=1, tol=1e-6)
                # If the new solution is better, replace the current solution
                if new_res.fun < res.fun:
                    return new_res.x
            return res.x
        else:
            return None

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the ProbabilisticDPHE algorithm
    dphe = ProbabilisticDPHE(budget=100, dim=10, mutation_prob=0.45)

    # Optimize the function
    result = dphe(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")