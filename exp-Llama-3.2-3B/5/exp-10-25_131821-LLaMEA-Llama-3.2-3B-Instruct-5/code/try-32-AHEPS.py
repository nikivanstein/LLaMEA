import numpy as np
from scipy.optimize import differential_evolution

class AHEPS:
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

        # Perform evolution strategies
        for _ in range(self.budget - len(elite_set)):
            # Evaluate population
            fitness = np.array([func(x) for x in population])

            # Select individuals for evolution strategies
            selected_individuals = np.random.choice(population, size=int(self.probability * self.budget), replace=False)

            # Perform evolution strategies
            new_population = np.array([self.evolve_strategy(selected_individual) for selected_individual in selected_individuals])

            # Update population and elite set
            population = np.concatenate((elite_set, new_population))
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

        # Return the best solution
        return np.min(func(population))

    def evolve_strategy(self, individual):
        # Perform evolution strategy
        new_individual = individual + np.random.uniform(-1.0, 1.0, size=self.dim)
        new_individual = np.clip(new_individual, self.search_space[0], self.search_space[1])
        return new_individual

# Example usage:
def func(x):
    return np.sum(x**2)

aheps = AHEPS(budget=100, dim=10)
best_solution = aheps(func)
print(f"Best solution: {best_solution}")