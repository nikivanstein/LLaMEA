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

    # Refine the strategy with probability 0.45
    refined_result = None
    for _ in range(10):
        individual = result.copy()
        for _ in range(int(np.random.uniform(0, 0.45) * len(result))):
            index = np.random.randint(len(individual))
            individual[index] = np.random.uniform(self.lower_bound, self.upper_bound)
        refined_result = dphe(individual)
        if refined_result is not None:
            break

    if refined_result is not None:
        print("Refined optimal solution:", refined_result)
    else:
        print("Failed to refine")

# Alternative implementation using a probabilistic approach
class DPHE Probabilistic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.probability = 0.45

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            return res.x
        else:
            return None

    def refine(self, func):
        refined_result = None
        for _ in range(10):
            individual = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            if np.random.rand() < self.probability:
                individual = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            refined_result = self(individual)
            if refined_result is not None:
                break

        return refined_result

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the DPHE algorithm
    dphe_probabilistic = DPHE Probabilistic(budget=100, dim=10)

    # Optimize the function
    result = dphe_probabilistic(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")

    # Refine the strategy with probability 0.45
    refined_result = dphe_probabilistic.refine(func)
    if refined_result is not None:
        print("Refined optimal solution:", refined_result)
    else:
        print("Failed to refine")