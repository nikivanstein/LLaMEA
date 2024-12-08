import numpy as np
from scipy.optimize import differential_evolution
import random

class HEAAdH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.mutator_ratio = 0.05

    def __call__(self, func):
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        for _ in range(self.budget - len(elite_set)):
            fitness = np.array([func(x) for x in population])
            new_population = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + 1, maxiter=1)

            elite_set = elite_set[np.argmin(fitness)]
            new_population = new_population[0:1]

            for i in range(len(new_population)):
                if random.random() < self.mutator_ratio:
                    new_population[i] = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)

            population = np.concatenate((elite_set, new_population))

        best_solution = np.min(func(population))
        return best_solution

# Example usage:
def func(x):
    return np.sum(x**2)

hea_adh = HEAAdH(budget=100, dim=10)
best_solution = hea_adh(func)
print(f"Best solution: {best_solution}")