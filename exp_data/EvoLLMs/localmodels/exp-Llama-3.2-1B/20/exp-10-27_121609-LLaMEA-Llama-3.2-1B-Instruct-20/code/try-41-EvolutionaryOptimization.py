import numpy as np
import random

class EvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []
        self.selection_probability = 0.2

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

            # Select individuals for mutation
            self.population_history.append(self.population)
            selected_indices = np.random.choice(self.population_size, self.population_size, replace=False, p=self.selection_probability)
            self.population = [self.population[i] for i in selected_indices]

            # Perform mutation
            for i in range(self.dim):
                if random.random() < 0.5:
                    new_x = self.population[i] + np.random.uniform(-5.0, 5.0) / 10.0
                    new_fitness = objective(new_x)
                    if new_fitness < self.fitnesses[i, new_x] + 1e-6:
                        self.fitnesses[i, new_x] = new_fitness
                        self.population[i] = new_x

        return self.fitnesses

# Example usage
def func(x):
    return x**2

nneo = EvolutionaryOptimization(10, 10)
print(nneo(__call__(func)))