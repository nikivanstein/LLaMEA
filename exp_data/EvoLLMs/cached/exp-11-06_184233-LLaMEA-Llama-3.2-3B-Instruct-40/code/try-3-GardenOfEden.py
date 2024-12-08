import numpy as np
import random
import operator

class GardenOfEden:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.fitness_values = None

    def __call__(self, func):
        self.fitness_values = np.zeros((self.population_size, self.dim))
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        for _ in range(self.budget):
            fitness_values = [func(x) for x in self.population]
            for i in range(self.population_size):
                self.fitness_values[i] = fitness_values[i]

            best_idx = np.argmax(self.fitness_values)
            best_individual = self.population[best_idx]

            # Select parents for crossover
            parents = []
            for _ in range(self.population_size // 2):
                parent1_idx = np.random.choice(self.population_size)
                parent2_idx = np.random.choice(self.population_size)
                while parent1_idx == parent2_idx:
                    parent2_idx = np.random.choice(self.population_size)
                parents.append(self.population[parent1_idx])
                parents.append(self.population[parent2_idx])

            # Perform crossover
            offspring = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = parents[0], parents[1]
                child = (parent1 + parent2) / 2
                offspring.append(child)

            # Perform mutation
            for i in range(self.population_size):
                if random.random() < self.mutation_rate:
                    mutation = np.random.uniform(-0.5, 0.5, size=self.dim)
                    offspring[i] += mutation

            # Replace worst individual with best offspring
            worst_idx = np.argmin(self.fitness_values)
            self.population[worst_idx] = offspring[np.argmin([func(x) for x in offspring])]

        return self.fitness_values[np.argmin(self.fitness_values)]

# Example usage:
def func(x):
    return sum([i**2 for i in x])

garden = GardenOfEden(budget=100, dim=10)
result = garden(func)
print(result)