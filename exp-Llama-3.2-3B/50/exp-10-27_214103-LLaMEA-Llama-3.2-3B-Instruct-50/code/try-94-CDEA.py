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
            new_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                if random.random() < 0.5:
                    parent1, parent2 = random.sample(self.population, 2)
                    child = np.concatenate((parent1, parent2[1:]))
                else:
                    child = np.random.choice(self.population)
                new_population[i] = child + np.random.normal(0, 1, self.dim)
                new_population[i] = np.clip(new_population[i], -5.0, 5.0)
            self.population = new_population

# Example usage:
def bbb_24_1(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2 + x[19]**2 + x[20]**2 + x[21]**2 + x[22]**2 + x[23]**2

budget = 100
dim = 20
cdea = CDEA(budget, dim)
results = cdea(bbb_24_1)