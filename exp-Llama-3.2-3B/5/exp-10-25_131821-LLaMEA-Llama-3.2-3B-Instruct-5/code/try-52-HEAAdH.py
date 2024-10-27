import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import truncnorm

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

        # Select elite individuals with adaptive hyper-elitism
        elite_indices = np.argsort(fitness[elite_set])[:int(self.budget * self.elitism_ratio)]
        elite_set = elite_set[elite_indices]

        # Perform adaptive hyper-elitism
        for i in range(len(elite_set)):
            # Calculate truncnorm parameters
            a = np.min(elite_set[i])
            b = np.max(elite_set[i])
            scale = (b - a) * self.probability
            loc = a + scale

            # Draw a random number from the truncnorm distribution
            rand = np.random.truncnorm(a, b, loc=loc, scale=scale)

            # Replace the current individual with the new one
            if rand > 0:
                elite_set[i] = elite_set[np.random.choice(len(elite_set), p=[0.5, 0.5])]

        # Return the best solution
        return np.min(func(population))

# Example usage:
def func(x):
    return np.sum(x**2)

hea_adh = HEAAdH(budget=100, dim=10)
best_solution = hea_adh(func)
print(f"Best solution: {best_solution}")