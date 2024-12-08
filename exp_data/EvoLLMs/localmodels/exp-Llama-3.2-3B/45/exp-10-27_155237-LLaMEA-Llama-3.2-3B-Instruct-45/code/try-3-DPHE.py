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

    # Refine the strategy with probability-based line search
    def refine_strategy(x, func, dphe):
        # Generate a new solution with probability 0.45
        if np.random.rand() < 0.45:
            new_x = x + np.random.uniform(-1, 1, size=len(x))
            new_x = np.clip(new_x, dphe.lower_bound, dphe.upper_bound)
            return new_x
        else:
            return x

    def evaluate_fitness(x, func, dphe):
        # Evaluate the fitness of the new solution
        new_x = refine_strategy(x, func, dphe)
        new_func_value = func(new_x)
        return new_func_value

    def hybrid_evolution(func, dphe):
        # Hybrid evolution with probability-based line search
        x = np.random.uniform(dphe.lower_bound, dphe.upper_bound, size=dphe.dim)
        x = np.clip(x, dphe.lower_bound, dphe.upper_bound)
        for _ in range(dphe.budget):
            new_x = refine_strategy(x, func, dphe)
            new_x = np.clip(new_x, dphe.lower_bound, dphe.upper_bound)
            new_func_value = evaluate_fitness(new_x, func, dphe)
            if new_func_value < func(x):
                x = new_x
        return x

    # Optimize the function with hybrid evolution
    result = hybrid_evolution(func, dphe)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")