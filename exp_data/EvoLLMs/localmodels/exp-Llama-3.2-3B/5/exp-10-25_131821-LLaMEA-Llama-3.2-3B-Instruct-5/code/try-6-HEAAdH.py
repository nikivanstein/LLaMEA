import numpy as np
from scipy.optimize import differential_evolution

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
            fitness = np.array([func(x) for x in population])
            new_population = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + 1, maxiter=1)

            elite_set = elite_set[np.random.choice(len(elite_set), int(self.budget * self.elitism_ratio), replace=False)]
            elite_set = np.vstack((elite_set, new_population[0:1]))

            # Replace some elite individuals with new ones
            to_replace = int(len(elite_set) * self.probability)
            elite_set = elite_set[np.random.choice(len(elite_set), to_replace, replace=False)]
            elite_set = elite_set[np.random.choice(len(elite_set), len(elite_set) - to_replace, replace=False)]

            population = np.concatenate((elite_set, new_population[0:1]))

        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

hea_adh = HEAAdH(budget=100, dim=10)
best_solution = hea_adh(func)
print(f"Best solution: {best_solution}")