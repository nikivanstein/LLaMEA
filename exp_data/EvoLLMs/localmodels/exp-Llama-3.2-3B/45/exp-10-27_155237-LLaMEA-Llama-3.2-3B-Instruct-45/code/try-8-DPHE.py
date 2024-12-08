import numpy as np
from scipy.optimize import differential_evolution

class DPHE:
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

    def mutate(self, individual):
        if np.random.rand() < self.probability:
            # Select two random parents
            parents = np.random.choice(individual, size=2, replace=False)

            # Perform crossover
            child = individual[(np.random.permutation(self.dim) < np.argsort(np.argsort(individual)))]

            # Perform mutation
            child += np.random.uniform(-0.1, 0.1, size=self.dim)

            # Ensure bounds are not exceeded
            child = np.clip(child, self.lower_bound, self.upper_bound)

            # Replace the worst individual with the child
            individuals = np.sort(individual)
            individuals[self.dim - 1] = child
            return individuals
        else:
            return individual

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