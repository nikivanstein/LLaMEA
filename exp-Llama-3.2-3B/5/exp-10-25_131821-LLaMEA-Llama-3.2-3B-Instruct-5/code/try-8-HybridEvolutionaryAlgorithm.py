import numpy as np
from scipy.optimize import differential_evolution
import random

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.differential_evolution_ratio = 0.5

    def __call__(self, func):
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        for _ in range(self.budget - len(elite_set)):
            fitness = np.array([func(x) for x in population])
            new_population = []

            for individual in elite_set:
                new_individual = individual + np.random.uniform(-1, 1, self.dim)
                new_individual = np.clip(new_individual, self.search_space[0], self.search_space[1])
                new_population.append(new_individual)

            # Perform differential evolution
            if random.random() < self.differential_evolution_ratio:
                new_population = differential_evolution(func, self.search_space, x0=new_population, popsize=len(new_population))

            # Update population and elite set
            population = np.concatenate((elite_set, new_population))
            elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Return the best solution
        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

hybrid_evolutionary_algorithm = HybridEvolutionaryAlgorithm(budget=100, dim=10)
best_solution = hybrid_evolutionary_algorithm(func)
print(f"Best solution: {best_solution}")