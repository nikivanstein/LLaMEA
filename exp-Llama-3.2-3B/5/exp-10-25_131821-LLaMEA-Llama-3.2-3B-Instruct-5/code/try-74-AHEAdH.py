import numpy as np
from scipy.optimize import differential_evolution

class AHEAdH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.differential_evolution_params = {
            'popsize': len(self.budget) * self.elitism_ratio + 1,
           'maxiter': 1,
            'x0': np.random.uniform(self.search_space[0], self.search_space[1], size=(len(self.budget) * self.elitism_ratio + 1, self.dim))
        }

    def __call__(self, func):
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        for _ in range(self.budget - len(elite_set)):
            fitness = np.array([func(x) for x in population])
            new_population = differential_evolution(func, self.search_space, **self.differential_evolution_params)

            population = np.concatenate((elite_set, new_population[0:1]))
            elite_set = population[:int(self.budget * self.elitism_ratio)]

        best_solution = np.min(func(population))
        return best_solution

# Example usage:
def func(x):
    return np.sum(x**2)

ahe_adh = AHEAdH(budget=100, dim=10)
best_solution = ahe_adh(func)
print(f"Best solution: {best_solution}")