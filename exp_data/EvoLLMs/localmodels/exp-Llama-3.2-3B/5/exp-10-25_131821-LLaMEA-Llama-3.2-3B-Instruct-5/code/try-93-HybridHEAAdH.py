import numpy as np
from scipy.optimize import differential_evolution

class HybridHEAAdH:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.elitism_ratio = 0.2
        self.stochastic_search_prob = 0.05

    def __call__(self, func):
        # Initialize population with random points
        population = np.random.uniform(self.search_space[0], self.search_space[1], size=(self.budget, self.dim))

        # Initialize elite set
        elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Perform differential evolution
        for _ in range(self.budget - len(elite_set)):
            # Evaluate population
            fitness = np.array([func(x) for x in population])

            # Perform stochastic search
            if np.random.rand() < self.stochastic_search_prob:
                new_population = self.stochastic_search(population, func)
            else:
                new_population = differential_evolution(func, self.search_space, x0=elite_set, popsize=len(elite_set) + 1, maxiter=1)

            # Update population and elite set
            population = np.concatenate((elite_set, new_population[0:1]))
            elite_set = population[:int(self.budget * self.elitism_ratio)]

        # Return the best solution
        return np.min(func(population))

    def stochastic_search(self, population, func):
        # Select a random individual
        individual = population[np.random.randint(0, len(population))]

        # Generate new individuals using mutation and crossover
        new_individuals = []
        for _ in range(10):
            # Perform mutation
            mutated_individual = individual + np.random.uniform(-1.0, 1.0, size=self.dim)

            # Perform crossover
            crossover_point = np.random.randint(0, self.dim)
            child = np.concatenate((mutated_individual[:crossover_point], individual[crossover_point:]))

            # Evaluate the child
            fitness = func(child)

            # Add the child to the new population
            new_individuals.append(child)

        # Return the best new individual
        return np.array(new_individuals)[np.argmin(func(new_individuals))]

# Example usage:
def func(x):
    return np.sum(x**2)

hybrid_hea_adh = HybridHEAAdH(budget=100, dim=10)
best_solution = hybrid_hea_adh(func)
print(f"Best solution: {best_solution}")