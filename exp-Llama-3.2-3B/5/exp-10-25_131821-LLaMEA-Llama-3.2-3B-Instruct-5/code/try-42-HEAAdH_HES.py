import numpy as np
from scipy.optimize import differential_evolution

class HEAAdH_HES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.hes_ratio = 0.05

    def __call__(self, func):
        # Initialize population with random points
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))

        # Initialize elite set
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Perform hybrid evolutionary strategy
        for _ in range(self.budget - len(elite_set)):
            # Evaluate population
            fitness = np.array([func(x) for x in population])

            # Perform differential evolution
            new_population = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + 1, maxiter=1)

            # Update population and elite set
            population = np.concatenate((elite_set, new_population[0:1]))
            elite_set = population[:int(self.budget * self.elitism_ratio)]

            # Perform evolutionary strategy
            if np.random.rand() < self.hes_ratio:
                # Select best individual
                best_individual = np.min(func(population))

                # Create a new population with the best individual
                new_population = np.tile(best_individual, (self.dim, 1))

                # Evaluate new population
                fitness = np.array([func(x) for x in new_population])

                # Replace the old population with the new one
                population = new_population

        # Return the best solution
        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

hea_adh_hes = HEAAdH_HES(budget=100, dim=10)
best_solution = hea_adh_hes(func)
print(f"Best solution: {best_solution}")