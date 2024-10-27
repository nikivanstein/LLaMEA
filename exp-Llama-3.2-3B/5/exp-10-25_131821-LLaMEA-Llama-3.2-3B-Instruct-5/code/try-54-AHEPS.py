import numpy as np
from scipy.optimize import differential_evolution

class AHEPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.probability = 0.05

    def __call__(self, func):
        # Initialize population with random points
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))

        # Initialize elite set
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Perform differential evolution
        for _ in range(self.budget - len(elite_set)):
            # Evaluate population
            fitness = np.array([func(x) for x in population])

            # Perform differential evolution
            new_population = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + 1, maxiter=1)

            # Update population and elite set with probabilistic elitism
            updated_population = []
            for i in range(len(elite_set)):
                if np.random.rand() < self.probability:
                    updated_population.append(new_population[0][i])
                else:
                    updated_population.append(elite_set[i])

            population = np.array(updated_population)

            # Update elite set
            elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Return the best solution
        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

aheps = AHEPS(budget=100, dim=10)
best_solution = aheps(func)
print(f"Best solution: {best_solution}")