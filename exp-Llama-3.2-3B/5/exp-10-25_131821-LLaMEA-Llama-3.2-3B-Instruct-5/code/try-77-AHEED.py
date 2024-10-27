import numpy as np
from scipy.optimize import differential_evolution

class AHEED:
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

        # Perform evolutionary strategies
        for _ in range(int(self.budget * self.probability)):
            # Select a random individual from the population
            individual = population[np.random.choice(self.budget, 1)[0]]

            # Perform evolutionary strategies
            new_individual = individual + np.random.uniform(-1, 1, size=self.dim)

            # Check if the new individual is within the search space
            if np.all(new_individual >= self.search_space[0]) and np.all(new_individual <= self.search_space[1]):
                # Replace the individual with the new one
                population[np.random.choice(self.budget, 1)[0]] = new_individual

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

# Example usage:
def func(x):
    return np.sum(x**2)

aheed = AHEED(budget=100, dim=10)
best_solution = aheed(func)
print(f"Best solution: {best_solution}")