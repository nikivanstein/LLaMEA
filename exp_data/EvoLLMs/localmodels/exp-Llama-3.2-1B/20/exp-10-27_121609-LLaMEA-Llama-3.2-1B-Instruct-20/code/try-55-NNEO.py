import numpy as np
import random

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

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

        return self.fitnesses

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []

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

            # Select new individuals based on probability
            selection_prob = 0.2
            selected_indices = np.random.choice(self.population_size, size=self.population_size, p=[1 - selection_prob, selection_prob])
            self.population = np.array([self.population[i] for i in selected_indices])

            # Mutate selected individuals
            for i in selected_indices:
                x = self.population[i]
                mutation_prob = 0.01
                mutated_x = x + np.random.normal(0, 1, self.dim) * mutation_prob
                self.population[i] = mutated_x

            # Evaluate fitness of new population
            new_population = self.__call__(func)
            self.population_history.append(new_population)

            # Replace old population with new population
            self.population = new_population

        return self.fitnesses

# Example usage:
def black_box_function(x):
    return np.sin(x) + 0.5 * x**2

ga = GeneticAlgorithm(budget=100, dim=10)
best_solution = ga(black_box_function)
print("Best solution:", best_solution)
print("Best fitness:", ga.fitnesses[best_solution])