import numpy as np
import random

class DifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness = np.zeros(self.population_size)
        self.best_individual = np.zeros(self.dim)
        self.best_fitness = -np.inf
        self.adaptive_mutation = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of the current population
            self.fitness = func(self.population)

            # Select the fittest individuals
            parents = np.argsort(self.fitness)[:-int(self.population_size/2):-1]
            self.population = self.population[parents]

            # Perform crossover and mutation
            offspring = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                if random.random() < 0.15:
                    # Adaptive mutation
                    mutation = np.random.uniform(-self.adaptive_mutation, self.adaptive_mutation, self.dim)
                    offspring[i] = self.population[i] + mutation
                else:
                    # Differential evolution
                    parent1, parent2 = random.sample(parents, 2)
                    r = random.random()
                    if r < 0.5:
                        offspring[i] = self.population[parent1] + (self.population[parent2] - self.population[parent1]) * r
                    else:
                        offspring[i] = self.population[parent2] + (self.population[parent1] - self.population[parent2]) * (1-r)

                # Clip the offspring to the search space
                offspring[i] = np.clip(offspring[i], -5.0, 5.0)

            # Calculate the fitness of the offspring
            self.fitness = func(offspring)

            # Update the population and the best individual
            self.population = np.concatenate((self.population, offspring))
            self.population = self.population[np.argsort(self.fitness)]
            self.best_individual = self.population[0]
            self.best_fitness = func(self.best_individual)

            # Check for convergence
            if self.best_fitness > self.best_fitness:
                break

        return self.best_individual, self.best_fitness

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = DifferentialEvolution(budget, dim)
best_individual, best_fitness = optimizer(func)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)