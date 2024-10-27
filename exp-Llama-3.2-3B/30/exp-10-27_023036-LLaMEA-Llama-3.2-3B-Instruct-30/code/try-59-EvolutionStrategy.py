import numpy as np
import random

class EvolutionStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = [self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.dim)]
        self.mutation_rate = 0.3

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the objective function
            x = np.array(self.population)
            f_x = func(x)
            # Update the population
            self.population = [self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.dim) for _ in range(len(x))]
            # Replace the worst individual with the best one
            self.population = [min(self.population, key=lambda x: f_x(x)) if f_x(x) < f_x(best) else best for best in self.population]
            # Refine the best individual with a probabilistic mutation
            if random.random() < self.mutation_rate:
                index = np.random.randint(0, len(self.population))
                self.population[index] = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.dim)
        return min(self.population, key=lambda x: func(x))

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

es = EvolutionStrategy(budget=100, dim=2)
best_x = es(func)
print(f"Best x: {best_x}")
print(f"Best f(x): {func(best_x)}")