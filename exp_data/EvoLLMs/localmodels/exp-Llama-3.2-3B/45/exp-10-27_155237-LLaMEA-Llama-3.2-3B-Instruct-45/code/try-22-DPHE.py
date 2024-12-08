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

    # Modify the DPHE algorithm to include probability-based mutation
    class ProbDPHE(DPHE):
        def __init__(self, budget, dim, prob):
            super().__init__(budget, dim)
            self.prob = prob

        def __call__(self, func):
            def neg_func(x):
                return -func(x)

            bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
            res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

            if res.success:
                # Apply probability-based mutation
                mutated_individual = np.copy(res.x)
                for i in range(self.dim):
                    if np.random.rand() < self.prob:
                        mutated_individual[i] = np.random.uniform(self.lower_bound, self.upper_bound)
                return mutated_individual
            else:
                return None

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the ProbDPHE algorithm
    prob_dphe = ProbDPHE(budget=100, dim=10, prob=0.45)

    # Optimize the function
    result = prob_dphe(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")