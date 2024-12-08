import numpy as np
from scipy.optimize import differential_evolution
import random

class HEAAdHRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.probability_refinement = 0.05

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
            updated_population = []
            for individual in population:
                if random.random() < self.probability_refinement:
                    # Refine individual by perturbing it randomly
                    refined_individual = individual + np.random.uniform(-0.1, 0.1, size=self.dim)
                    refined_individual = np.clip(refined_individual, self.search_space[0], self.search_space[1])
                    updated_population.append(refined_individual)
                else:
                    updated_population.append(new_population[0])

            population = updated_population

        # Return the best solution
        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

hea_adh_refined = HEAAdHRefined(budget=100, dim=10)
best_solution = hea_adh_refined(func)
print(f"Best solution: {best_solution}")