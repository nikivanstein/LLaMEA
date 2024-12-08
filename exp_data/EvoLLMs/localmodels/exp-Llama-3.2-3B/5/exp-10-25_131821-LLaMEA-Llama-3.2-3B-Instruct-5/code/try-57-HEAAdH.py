import numpy as np
from scipy.optimize import differential_evolution
import random

class HEAAdH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2

    def __call__(self, func):
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        for _ in range(self.budget - len(elite_set)):
            fitness = np.array([func(x) for x in population])
            new_population = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + 1, maxiter=1)

            # Randomly replace 20% of elite set with new population
            replace_indices = np.random.choice(len(elite_set), size=int(0.2 * len(elite_set)), replace=False)
            new_population = new_population[0:1]
            for i in replace_indices:
                elite_set[i] = new_population[0]

            population = np.concatenate((elite_set, new_population[0:1]))

        best_solution = np.min(func(population))
        return best_solution

# Example usage:
def func(x):
    return np.sum(x**2)

hea_adh = HEAAdH(budget=100, dim=10)
best_solution = hea_adh(func)
print(f"Best solution: {best_solution}")