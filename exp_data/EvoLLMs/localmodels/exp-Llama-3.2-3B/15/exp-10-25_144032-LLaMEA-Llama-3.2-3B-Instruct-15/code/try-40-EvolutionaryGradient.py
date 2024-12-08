import numpy as np
import random

class EvolutionaryGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness = np.zeros(self.population_size)
        self.best_individual = np.zeros(self.dim)
        self.best_fitness = -np.inf
        self.refine_probability = 0.15

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
                parent1, parent2 = random.sample(parents, 2)
                offspring[i] = (self.population[parent1] + self.population[parent2]) / 2 + np.random.uniform(-0.1, 0.1, self.dim)
                offspring[i] = np.clip(offspring[i], -5.0, 5.0)

            # Calculate the gradient of the fitness function
            gradient = np.zeros((self.dim,))
            for i in range(self.population_size):
                gradient += (func(offspring[i]) - func(self.population[i])) * (offspring[i] - self.population[i]) / (self.population_size - 1)

            # Refine the population with a probability of 0.15
            if random.random() < self.refine_probability:
                for i in range(self.population_size):
                    # Randomly select an individual to refine
                    if random.random() < 0.5:
                        # Refine the individual by adding a small mutation
                        mutation = np.random.uniform(-0.01, 0.01, self.dim)
                        offspring[i] += mutation
                        offspring[i] = np.clip(offspring[i], -5.0, 5.0)
                    # Refine the individual by using the gradient
                    else:
                        # Calculate the gradient of the fitness function at the current individual
                        gradient_at_individual = np.zeros((self.dim,))
                        for j in range(self.dim):
                            gradient_at_individual[j] = (func(offspring[i]) - func(self.population[i])) / (offspring[i][j] - self.population[i][j])
                        # Refine the individual by moving it in the direction of the negative gradient
                        offspring[i] -= 0.1 * gradient_at_individual

            # Update the population and the best individual
            self.population = np.concatenate((self.population, offspring))
            self.fitness = func(self.population)
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
optimizer = EvolutionaryGradient(budget, dim)
best_individual, best_fitness = optimizer(func)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)