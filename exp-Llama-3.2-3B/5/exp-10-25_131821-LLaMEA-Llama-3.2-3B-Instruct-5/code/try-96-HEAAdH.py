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
        # Initialize population with random points
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))

        # Initialize elite set
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Perform differential evolution
        for _ in range(self.budget - len(elite_set)):
            # Evaluate population
            fitness = np.array([func(x) for x in population])

            # Perform differential evolution
            new_population = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + 1, maxiter=1)

            # Update population and elite set
            population = np.concatenate((elite_set, new_population[0:1]))
            elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Select top 20% of population for crossover
        top_20 = np.percentile(population, 20)
        new_population = top_20[:int(self.budget * self.probability)]

        # Perform crossover
        offspring = np.zeros((len(new_population), self.dim))
        for i in range(len(new_population)):
            parent1, parent2 = np.random.choice(population, size=2, replace=False)
            offspring[i] = (parent1 + parent2) / 2

        # Add new offspring to population
        population = np.concatenate((population, offspring))

        # Evaluate population
        fitness = np.array([func(x) for x in population])

        # Perform mutation
        mutation_rate = 0.1
        mutated_population = population.copy()
        for i in range(len(population)):
            if np.random.rand() < mutation_rate:
                mutated_population[i] += np.random.uniform(-1, 1, self.dim)

        # Update population
        population = mutated_population

        # Return the best solution
        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

hea_adh = HEAAdH(budget=100, dim=10)
best_solution = hea_adh(func)
print(f"Best solution: {best_solution}")