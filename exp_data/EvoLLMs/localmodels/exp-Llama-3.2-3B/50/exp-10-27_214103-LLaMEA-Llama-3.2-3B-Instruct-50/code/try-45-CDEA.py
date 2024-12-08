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

            # Perform probability-based crossover and mutation
            self.population = self.probability_based_crossover(self.population)
            self.population = self.probability_based_mutation(self.population)

    def probability_based_crossover(self, population):
        offspring = []
        for _ in range(len(population)):
            parent1, parent2 = random.sample(population, 2)
            if random.random() < 0.5:
                child = np.concatenate((parent1, parent2[1:]))
            else:
                child = np.concatenate((parent2, parent1[1:]))
            offspring.append(child)
        return np.array(offspring)

    def probability_based_mutation(self, population):
        mutated_population = population + np.random.normal(0, 1, population.shape)
        mutated_population = np.clip(mutated_population, -5.0, 5.0)
        mutated_indices = np.where(np.random.rand(len(population)) < 0.5)[0]
        mutated_population[mutated_indices] = np.random.uniform(-5.0, 5.0, len(mutated_indices))
        return mutated_population

# Example usage
def func(x):
    return x[0]**2 + x[1]**2 + x[2]**2

bde = CDEA(100, 3)
bde(func)