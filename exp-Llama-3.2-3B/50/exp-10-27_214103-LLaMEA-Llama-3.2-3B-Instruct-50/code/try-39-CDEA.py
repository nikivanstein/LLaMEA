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
        self.mutation_prob = 0.5
        self.crossover_prob = 0.5

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

            # Perform crossover and mutation
            offspring = []
            for _ in range(len(self.population)):
                parent1, parent2 = random.sample(self.population, 2)
                if random.random() < self.crossover_prob:
                    child = np.concatenate((parent1, parent2[1:]))
                else:
                    child = np.concatenate((parent1, parent2))
                if random.random() < self.mutation_prob:
                    child = child + np.random.normal(0, 1, child.shape)
                child = np.clip(child, -5.0, 5.0)
                offspring.append(child)
            self.population = np.array(offspring)

# Test the algorithm
def test_func(x):
    return x[0]**2 + x[1]**2

cdea = CDEA(100, 2)
cdea()