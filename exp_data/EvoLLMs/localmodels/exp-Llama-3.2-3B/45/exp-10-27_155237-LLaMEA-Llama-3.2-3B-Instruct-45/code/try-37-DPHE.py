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
class DPHE_Mutation:
    def __init__(self, budget, dim, mutation_prob=0.45):
        self.budget = budget
        self.dim = dim
        self.mutation_prob = mutation_prob
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func, current_individual):
        def neg_func(x):
            return -func(x)

        bounds = [(self.lower_bound, self.upper_bound) for _ in range(self.dim)]
        res = differential_evolution(func, bounds, x0=current_individual, maxiter=self.budget, tol=1e-6)

        if res.success:
            new_individual = res.x
        else:
            new_individual = current_individual

        # Apply mutation with probability
        if np.random.rand() < self.mutation_prob:
            new_individual = self.mutate(new_individual)

        return new_individual

    def mutate(self, individual):
        new_individual = individual.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                new_individual[i] += np.random.uniform(-1, 1)
                new_individual[i] = np.clip(new_individual[i], self.lower_bound, self.upper_bound)
        return new_individual

# Example usage:
if __name__ == "__main__":
    # Define a sample black box function
    def func(x):
        return np.sum(x**2)

    # Initialize the DPHE algorithm with mutation
    dphe_mutation = DPHE_Mutation(budget=100, dim=10, mutation_prob=0.45)

    # Optimize the function
    current_individual = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
    for _ in range(10):
        result = dphe_mutation(func, current_individual)
        print("New individual:", result)
        current_individual = result