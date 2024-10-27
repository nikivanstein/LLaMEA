import numpy as np
from scipy.optimize import differential_evolution
import random

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.tuning_ratio = 0.05

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

            # Hyper-parameter tuning
            for i in range(int(self.budget * self.tuning_ratio)):
                # Randomly select an individual
                individual = population[np.random.choice(len(population), 1)[0]]

                # Randomly select a hyper-parameter
                hyper_param = random.choice(['learning_rate', 'population_size'])

                # Randomly select a value for the hyper-parameter
                if hyper_param == 'learning_rate':
                    new_value = np.random.uniform(0.01, 0.1)
                else:
                    new_value = np.random.randint(10, 100)

                # Update the individual
                if hyper_param == 'learning_rate':
                    individual[0] = individual[0] + new_value
                else:
                    individual[1] = individual[1] + new_value

                # Evaluate the individual
                fitness = np.array([func(x) for x in population])
                population = np.concatenate((population, [individual]))

        # Return the best solution
        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

hybrid_ea = HybridEvolutionaryAlgorithm(budget=100, dim=10)
best_solution = hybrid_ea(func)
print(f"Best solution: {best_solution}")