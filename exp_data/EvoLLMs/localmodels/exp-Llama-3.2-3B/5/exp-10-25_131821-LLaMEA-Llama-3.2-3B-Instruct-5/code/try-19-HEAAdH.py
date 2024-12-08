import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import randint

class HEAAdH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.param_grid = {
            'npop': randint(10, 100),
           'maxiter': randint(10, 100),
            'x0': randint(10, 100),
            'popsize': randint(10, 100)
        }

    def __call__(self, func):
        # Initialize population with random points
        param_grid = {k: randint(10, 100) for k in self.param_grid.keys()}
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))

        # Initialize elite set
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Perform differential evolution
        for _ in range(self.budget - len(elite_set)):
            # Evaluate population
            fitness = np.array([func(x) for x in population])

            # Perform hyperparameter tuning
            param_grid = self.tune_hyperparameters(fitness, param_grid)

            # Perform differential evolution with hyperparameter tuning
            new_population = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + 1, **param_grid, maxiter=1)

            # Update population and elite set
            population = np.concatenate((elite_set, new_population[0:1]))
            elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Return the best solution
        return np.min(func(population))

    def tune_hyperparameters(self, fitness, param_grid):
        best_param = min(fitness, key=fitness.item)
        param_grid['npop'] = best_param['npop']
        param_grid['maxiter'] = best_param['maxiter']
        param_grid['x0'] = best_param['x0']
        param_grid['popsize'] = best_param['popsize']
        return param_grid

# Example usage:
def func(x):
    return np.sum(x**2)

hea_adh = HEAAdH(budget=100, dim=10)
best_solution = hea_adh(func)
print(f"Best solution: {best_solution}")