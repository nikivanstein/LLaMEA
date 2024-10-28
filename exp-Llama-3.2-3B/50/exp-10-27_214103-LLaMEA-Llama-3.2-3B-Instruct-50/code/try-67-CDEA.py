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
            mutated_population = self.mutate(self.population)
            crossovered_population = self.crossover(mutated_population, self.population, 0.5)
            self.population = np.concatenate((crossovered_population, self.population[:self.population_size-self.population.shape[0]]))

    def crossover(self, population1, population2, prob):
        # Perform single-point crossover with probability
        offspring = []
        for _ in range(len(population1)):
            if random.random() < prob:
                parent1, parent2 = random.sample(population1, 2)
                child = np.concatenate((parent1, parent2[1:]))
                offspring.append(child)
            else:
                offspring.append(random.choice(population1))
        return np.array(offspring)

    def mutate(self, population):
        # Perform Gaussian mutation with probability 0.5
        mutated_population = population + np.random.normal(0, 1, population.shape)
        return np.clip(mutated_population, -5.0, 5.0)

# Example usage
def func(x):
    return x[0]**2 + x[1]**2 + x[2]**2

CDEA_obj = CDEA(100, 3)
CDEA_obj()