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

# Novel mutation strategy with probability-based mutation
class NovelDPHE(DPHE):
    def __init__(self, budget, dim, mutation_prob=0.45):
        super().__init__(budget, dim)
        self.mutation_prob = mutation_prob

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            # Apply probability-based mutation
            if np.random.rand() < self.mutation_prob:
                # Randomly select two individuals
                ind1, ind2 = np.random.choice(res.x, size=2, replace=False)
                # Perturb the first individual
                perturbed_ind1 = ind1 + np.random.uniform(-0.5, 0.5, size=self.dim)
                # Perturb the second individual
                perturbed_ind2 = ind2 + np.random.uniform(-0.5, 0.5, size=self.dim)
                # Replace the worst individual with the perturbed individual
                res.x[res.x == min(res.x)] = perturbed_ind1
                res.x[res.x == max(res.x)] = perturbed_ind2
            return res.x
        else:
            return None

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the novel DPHE algorithm
    novel_dphe = NovelDPHE(budget=100, dim=10, mutation_prob=0.45)

    # Optimize the function
    result = novel_dphe(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")