import numpy as np
from scipy.optimize import differential_evolution

class HEAAdH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.population_size = int(self.budget * (1 - self.elitism_ratio))

    def __call__(self, func):
        # Initialize population with random points
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.population_size, self.dim))

        # Initialize elite set
        elite_set = population[:int(self.population_size * self.elitism_ratio)]

        # Perform differential evolution
        for _ in range(self.budget - len(elite_set)):
            # Evaluate population
            fitness = np.array([func(x) for x in population])

            # Perform differential evolution
            new_population = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + 1, maxiter=1)

            # Update population and elite set
            population = np.concatenate((elite_set, new_population[0:1]))
            elite_set = population[:int(self.population_size * self.elitism_ratio)]

        # Select elite and perform evolutionary strategy
        elite = np.random.choice(population, size=int(self.population_size * self.elitism_ratio), replace=False)
        new_population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.population_size, self.dim))
        for i in range(self.population_size):
            if np.random.rand() < 0.05:
                new_population[i] = elite[np.random.randint(0, len(elite))]
            else:
                new_population[i] = np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim)

        # Update population and elite set
        population = np.concatenate((elite, new_population))
        elite_set = population[:int(self.population_size * self.elitism_ratio)]

        # Return the best solution
        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

hea_adh = HEAAdH(budget=100, dim=10)
best_solution = hea_adh(func)
print(f"Best solution: {best_solution}")