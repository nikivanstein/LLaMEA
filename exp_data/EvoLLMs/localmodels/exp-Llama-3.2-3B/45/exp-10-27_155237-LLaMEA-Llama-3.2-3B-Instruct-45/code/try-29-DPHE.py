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

# Modified DPHE algorithm with probabilistic mutation
class DPHE_Probabilistic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.p_mutate = 0.45  # probability of mutation

    def __call__(self, func):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), maxiter=self.budget, tol=1e-6)

        if res.success:
            # Refine the solution with probabilistic mutation
            if np.random.rand() < self.p_mutate:
                new_individual = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
                new_individual = self.f(new_individual, func)
                res.x = new_individual
            return res.x
        else:
            return None

    def f(self, individual, func):
        # Evaluate the fitness of the individual
        fitness = func(individual)
        # Perturb the individual with a small mutation
        mutated_individual = individual + np.random.uniform(-0.1, 0.1, size=self.dim)
        mutated_fitness = func(mutated_individual)
        # Select the individual with better fitness
        if mutated_fitness < fitness:
            return mutated_individual
        else:
            return individual

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the DPHE algorithm
    dphe_probabilistic = DPHE_Probabilistic(budget=100, dim=10)

    # Optimize the function
    result = dphe_probabilistic(func)

    # Print the result
    if result is not None:
        print("Optimal solution:", result)
    else:
        print("Failed to converge")