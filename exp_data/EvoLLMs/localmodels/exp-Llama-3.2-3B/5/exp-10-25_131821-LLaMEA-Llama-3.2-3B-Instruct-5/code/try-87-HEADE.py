import numpy as np
from scipy.optimize import differential_evolution

class HEADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.differential_evolution_params = {
            'x0': np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget - int(self.budget * self.elitism_ratio), self.dim)),
            'popsize': int(self.budget - int(self.budget * self.elitism_ratio)) + 1,
           'maxiter': 1,
           'method': 'DE/rand/1/bin',
            'bounds': self.search_space,
            'xtol': 1e-6,
            'ftol': 1e-6,
        }

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
            new_population = differential_evolution(func, self.search_space, **self.differential_evolution_params)

            # Update population and elite set
            population = np.concatenate((elite_set, new_population[0:1]))
            elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Return the best solution
        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

hea_de = HEADE(budget=100, dim=10)
best_solution = hea_de(func)
print(f"Best solution: {best_solution}")