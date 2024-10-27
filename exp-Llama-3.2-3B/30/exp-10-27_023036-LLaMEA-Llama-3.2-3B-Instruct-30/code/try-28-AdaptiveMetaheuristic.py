import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.fitness = np.zeros(budget)
        self.score = np.inf

    def __call__(self, func):
        for i in range(self.budget):
            # Evaluate the function at each point in the population
            self.fitness[i] = func(self.population[i])

            # Calculate the fitness score for each point
            self.score = np.min(self.score, np.min(self.fitness))

            # Apply mutation and crossover to the population
            self.population[i] = self.mutate(self.population[i])
            self.population[i] = self.crossover(self.population[i])

        # Select the best point in the population
        self.population = self.population[np.argmin(self.fitness)]

    def mutate(self, point):
        # Apply mutation with probability 0.3
        if np.random.rand() < 0.3:
            point = point + np.random.uniform(-1.0, 1.0, self.dim)
        return point

    def crossover(self, point):
        # Apply crossover with probability 0.3
        if np.random.rand() < 0.3:
            point1 = point + np.random.uniform(-1.0, 1.0, self.dim)
            point2 = point - np.random.uniform(-1.0, 1.0, self.dim)
            point = (point1 + point2) / 2
        return point

# Test the algorithm
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2

algorithm = AdaptiveMetaheuristic(budget, dim)
best_point = algorithm(func)

# Print the best point found
print("Best point:", best_point)
print("Fitness:", func(best_point))