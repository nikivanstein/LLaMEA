import numpy as np
from scipy.optimize import differential_evolution
import random

class HEAAdH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.probability = 0.05

    def __call__(self, func):
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        for _ in range(self.budget - len(elite_set)):
            new_population = []
            for individual in elite_set:
                if random.random() < self.probability:
                    new_individual = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)
                    new_population.append(new_individual)
                else:
                    new_population.append(individual)

            fitness = np.array([func(x) for x in new_population])
            new_population = differential_evolution(func, self.search_space, x0=new_population, popsize=len(new_population))

            population = np.concatenate((elite_set, new_population))
            elite_set = population[:int(self.budget * self.elitism_ratio)]

        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

hea_adh = HEAAdH(budget=100, dim=10)
best_solution = hea_adh(func)
print(f"Best solution: {best_solution}")