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
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(int(self.budget * (1 - self.elitism_ratio)), self.dim))
        elite_set = np.random.choice(population, size=int(self.budget * self.elitism_ratio), replace=False)

        for _ in range(int(self.budget - len(elite_set))):
            new_population = np.concatenate((elite_set, differential_evolution(func, self.search_space, x0=np.random.uniform(self.search_space[0], self.search_space[1], size=(len(elite_set) + 1, self.dim)), popsize=len(elite_set) + 1, maxiter=1)[0:1]))

            if np.random.rand() < self.probability:
                elite_set = np.random.choice(new_population, size=int(self.budget * self.elitism_ratio), replace=False)

            population = np.concatenate((elite_set, new_population[1:]))

        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

hea_adh = HEAAdH(budget=100, dim=10)
best_solution = hea_adh(func)
print(f"Best solution: {best_solution}")