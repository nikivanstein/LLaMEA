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

# Novel "Differential Perturbation and Hybrid Evolution" (DPHE) algorithm with probabilistic mutation
class ProbabilisticDPHE(DPHE):
    def __init__(self, budget, dim, mutation_prob=0.45):
        super().__init__(budget, dim)
        self.mutation_prob = mutation_prob

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            # Apply probabilistic mutation to refine the solution
            if np.random.rand() < self.mutation_prob:
                new_individual = np.copy(res.x)
                for i in range(self.dim):
                    if np.random.rand() < 0.5:
                        new_individual[i] += np.random.uniform(-1, 1)
                    elif np.random.rand() < 0.5:
                        new_individual[i] -= np.random.uniform(-1, 1)
                res = differential_evolution(func, bounds, x0=new_individual, maxiter=1, tol=1e-6)
            return res.x
        else:
            return None

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the ProbabilisticDPHE algorithm
    prob_dphe = ProbabilisticDPHE(budget=100, dim=10)

    # Optimize the function
    result = prob_dphe(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")