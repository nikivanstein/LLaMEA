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
            mutated_population = np.copy(self.population)
            for i in range(self.population_size):
                if random.random() < 0.5:
                    # Perform single-point crossover
                    parent1, parent2 = random.sample(self.population, 2)
                    child = np.concatenate((parent1, parent2[1:]))
                    mutated_population[i] = child
                else:
                    # Perform Gaussian mutation
                    mutated_population[i] = self.population[i] + np.random.normal(0, 1, self.dim)
                    mutated_population[i] = np.clip(mutated_population[i], -5.0, 5.0)

    def crossover(self, population):
        # Perform single-point crossover
        offspring = []
        for _ in range(len(population)):
            parent1, parent2 = random.sample(population, 2)
            child = np.concatenate((parent1, parent2[1:]))
            offspring.append(child)
        return np.array(offspring)

    def mutate(self, population):
        # Perform Gaussian mutation
        mutated_population = population + np.random.normal(0, 1, population.shape)
        return np.clip(mutated_population, -5.0, 5.0)

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2 + x[2]**2

cdea = CDEA(100, 3)
cdea(__call__, func)