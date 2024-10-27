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

def mutation(individual, logger):
    if np.random.rand() < 0.45:
        mutation_rate = np.random.uniform(0.1, 0.5)
        mutated_individual = individual + mutation_rate * np.random.normal(0, 1, size=individual.shape)
        mutated_individual = np.clip(mutated_individual, self.lower_bound, self.upper_bound)
    else:
        mutated_individual = individual
    return mutated_individual

def hybrid_evolution(func, population, logger):
    for _ in range(10):  # number of generations
        new_population = []
        for individual in population:
            new_individual = mutation(individual, logger)
            new_population.append(new_individual)
        population = new_population
    return population

def optimize_function(func, algorithm, budget, dim):
    algorithm = algorithm()
    population = [np.random.uniform(-5.0, 5.0, size=dim) for _ in range(100)]
    for _ in range(budget):
        population = hybrid_evolution(func, population, None)
        best_individual = min(population, key=lambda x: func(x))
        logger = {"aucs": [func(best_individual)]}
        np.save("currentexp/aucs-" + algorithm.__name__ + "-" + str(len(logger["aucs"])) + ".npy", logger["aucs"])
    return algorithm()

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