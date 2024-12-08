import numpy as np
import random

class PAEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitness_values = []

    def __call__(self, func):
        if len(self.population) < self.budget:
            # Initialize population with random points
            self.population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.budget)]
            self.fitness_values = [func(point) for point in self.population]
        else:
            # Replace old population with new ones using probability
            new_population = []
            for _ in range(self.budget):
                if random.random() < 0.3:
                    # Replace with new random point
                    new_population.append(np.random.uniform(-5.0, 5.0, self.dim))
                else:
                    # Replace with best individual
                    new_population.append(self.population[np.argmin(self.fitness_values)])
            self.population = new_population
            self.fitness_values = [func(point) for point in self.population]

        # Evaluate fitness values
        self.fitness_values = [func(point) for point in self.population]

        # Select best individual
        best_index = np.argmin(self.fitness_values)
        self.population = [self.population[best_index]]
        self.fitness_values = [self.fitness_values[best_index]]

    def get_best_solution(self):
        return self.population[0]

# Example usage
def func(x):
    return sum(i**2 for i in x)

paea = PAEA(budget=100, dim=10)
paea(func)
print(paea.get_best_solution())