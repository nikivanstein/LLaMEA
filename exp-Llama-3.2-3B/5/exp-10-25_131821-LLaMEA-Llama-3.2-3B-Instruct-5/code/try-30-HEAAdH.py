import numpy as np
from scipy.optimize import differential_evolution
from copy import deepcopy

class HEAAdH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.mutate_prob = 0.05
        self.crossover_prob = 0.7

    def __call__(self, func):
        # Initialize population with random points
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))

        # Initialize elite set
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Perform evolutionary strategy
        for _ in range(self.budget - len(elite_set)):
            # Evaluate population
            fitness = np.array([func(x) for x in population])

            # Select parents for crossover
            parents = np.random.choice(population, size=len(population), replace=False, p=fitness/np.max(fitness))

            # Perform crossover
            offspring = []
            for _ in range(len(population) - len(elite_set)):
                parent1, parent2 = parents[np.random.choice(len(parents), size=2, replace=False, p=np.array([self.crossover_prob]*2 + np.array([1-self.crossover_prob]*2)))]
                child = np.mean([parent1, parent2], axis=0)
                if np.random.rand() < self.mutate_prob:
                    child += np.random.uniform(-1, 1, size=self.dim)
                offspring.append(child)

            # Update population and elite set
            population = np.concatenate((elite_set, offspring))

            # Perform differential evolution on elite set
            new_elite_set = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + 1, maxiter=1)
            elite_set = new_elite_set[0:1]

        # Return the best solution
        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

hea_adh = HEAAdH(budget=100, dim=10)
best_solution = hea_adh(func)
print(f"Best solution: {best_solution}")