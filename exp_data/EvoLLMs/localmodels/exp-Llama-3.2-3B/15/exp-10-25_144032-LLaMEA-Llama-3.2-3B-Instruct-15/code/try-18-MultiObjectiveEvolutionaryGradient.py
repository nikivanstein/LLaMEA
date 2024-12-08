import numpy as np
import random

class MultiObjectiveEvolutionaryGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness = np.zeros((self.population_size, 2))  # store fitness for each objective
        self.best_individual = np.zeros((self.dim, 2))  # store best individual for each objective
        self.best_fitness = np.inf
        self.objective_weights = np.array([0.5, 0.5])  # weights for each objective

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of the current population
            self.fitness = func(self.population)

            # Select the fittest individuals
            parents = np.argsort(np.sum(self.fitness, axis=1))[:-int(self.population_size/2):-1]
            self.population = self.population[parents]

            # Perform crossover and mutation
            offspring = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                offspring[i] = (self.population[parent1] + self.population[parent2]) / 2 + np.random.uniform(-0.1, 0.1, self.dim)
                offspring[i] = np.clip(offspring[i], -5.0, 5.0)

            # Calculate the gradient of the fitness function
            gradient = np.zeros((self.dim, 2))
            for i in range(self.population_size):
                gradient += (func(offspring[i]) - func(self.population[i])) * (offspring[i] - self.population[i]) / (self.population_size - 1)

            # Update the population and the best individual
            self.population = np.concatenate((self.population, offspring))
            self.fitness = func(self.population)
            self.population = self.population[np.argsort(np.sum(self.fitness, axis=1))]
            self.best_individual = self.population[0]
            self.best_fitness = np.sum(func(self.best_individual), axis=1)

            # Refine the strategy by changing individual lines with probability 0.15
            refine_probability = 0.15
            for i in range(self.population_size):
                if np.random.rand() < refine_probability:
                    # Select an objective randomly
                    objective_index = np.random.randint(0, 2)
                    # Update the individual's value for the selected objective
                    self.population[i, objective_index] = self.population[i, objective_index] + np.random.uniform(-0.1, 0.1)
                    self.population[i] = np.clip(self.population[i], -5.0, 5.0)

            # Check for convergence
            if np.sum(self.best_fitness) < self.best_fitness:
                break

        return self.best_individual, np.sum(func(self.best_individual))

# Example usage:
def func(x):
    return np.sum(x**2), np.sum(x**3)  # two objectives

budget = 100
dim = 10
optimizer = MultiObjectiveEvolutionaryGradient(budget, dim)
best_individual, best_fitness = optimizer(func)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)