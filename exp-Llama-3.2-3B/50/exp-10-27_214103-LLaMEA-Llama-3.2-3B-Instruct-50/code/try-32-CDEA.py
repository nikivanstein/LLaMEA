import numpy as np
import random

class CDEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crowd_size = 10
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.crowd = np.random.uniform(-5.0, 5.0, (self.crowd_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the population
            population_values = np.array([func(x) for x in self.population])

            # Evaluate the crowd
            crowd_values = np.array([func(x) for x in self.crowd])

            # Select the best individuals
            best_indices = np.argsort(population_values)[:, -self.crowd_size:]
            best_crowd_values = crowd_values[best_indices]

            # Select the worst individuals
            worst_indices = np.argsort(population_values)[:, :self.crowd_size]
            worst_population_values = population_values[worst_indices]

            # Update the population
            self.population = np.concatenate((best_crowd_values, worst_population_values))

            # Update the crowd
            self.crowd = self.population[:self.crowd_size]

            # Perform probabilistic crossover and mutation
            new_population = []
            for i in range(self.population_size):
                parent1, parent2 = random.sample(self.population, 2)
                if random.random() < 0.5:
                    child = np.concatenate((parent1, parent2[1:]))
                else:
                    child = np.concatenate((parent2, parent1[1:]))
                new_population.append(child)
            self.population = np.array(new_population)

            # Perform probabilistic mutation
            mutated_population = self.population + np.random.normal(0, 1, self.population.shape)
            mutated_population = np.clip(mutated_population, -5.0, 5.0)
            self.population = mutated_population

# Usage
def bbb_func(x):
    return np.sum(x**2)

cdea = CDEA(50, 10)
cdea('bbb_func')