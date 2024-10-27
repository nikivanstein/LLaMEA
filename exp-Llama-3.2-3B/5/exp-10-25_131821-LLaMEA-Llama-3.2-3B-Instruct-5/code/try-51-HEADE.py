import numpy as np
from scipy.optimize import differential_evolution

class HEADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.differential_evolution_params = {'maxiter': 1, 'popsize': len(self.budget) * 0.8}

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
            new_population = differential_evolution(func, self.search_space, x0=elite_set, **self.differential_evolution_params)

            # Update population and elite set
            population = np.concatenate((elite_set, new_population[0:1]))
            elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Perform evolutionary strategies
        for i in range(len(population)):
            # Perform mutation
            mutation = np.random.uniform(-1.0, 1.0, size=self.dim)
            population[i] += mutation

            # Perform crossover
            if np.random.rand() < 0.05:
                crossover = np.random.choice(population, size=self.dim, replace=False)
                population[i] = (population[i] + crossover) / 2

        # Return the best solution
        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

hea_de = HEADE(budget=100, dim=10)
best_solution = hea_de(func)
print(f"Best solution: {best_solution}")