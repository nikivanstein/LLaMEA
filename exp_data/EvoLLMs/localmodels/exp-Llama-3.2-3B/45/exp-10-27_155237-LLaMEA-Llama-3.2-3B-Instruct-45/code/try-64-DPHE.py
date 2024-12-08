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

# Probability-based mutation and exploration
    def mutate(individual):
        new_individual = individual.copy()
        if np.random.rand() < self.probability:
            new_individual += np.random.uniform(-1, 1, size=self.dim)
            new_individual = np.clip(new_individual, self.lower_bound, self.upper_bound)
        return new_individual

    def evaluate_fitness(individual, logger):
        # Evaluate the fitness of the individual
        fitness = func(individual)
        logger.log(fitness)
        return fitness

    def hybrid_evolution(individual, logger):
        # Hybrid evolution using mutation and exploration
        new_individual = mutate(individual)
        fitness = evaluate_fitness(new_individual, logger)
        return new_individual, fitness

    # Main loop
    for i in range(100):
        individual = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
        new_individual, fitness = hybrid_evolution(individual, logger)
        logger.log(f"Generation {i+1}, Individual: {new_individual}, Fitness: {fitness}")