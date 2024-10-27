# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random

class NNEO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            """Randomly mutate an individual"""
            new_individual = individual.copy()
            for i in range(self.dim):
                if random.random() < 0.2:
                    new_individual[i] += random.uniform(-0.1, 0.1)
            return new_individual

        def evaluate_fitness(individual):
            """Evaluate the fitness of an individual"""
            fitness = objective(individual)
            if fitness < self.fitnesses[individual, :].max() + 1e-6:
                self.fitnesses[individual, :] = fitness
            return fitness

        def bounds_eval(individual):
            """Evaluate the bounds of an individual"""
            return bounds(individual)

        for _ in range(self.budget):
            for i in range(self.population_size):
                new_individual = mutate(self.population[i])
                fitness = evaluate_fitness(new_individual)
                if fitness < self.fitnesses[i, new_individual] + 1e-6:
                    self.fitnesses[i, new_individual] = fitness
                    self.population[i] = new_individual

        return self.fitnesses

# One-line description with main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# This algorithm uses a novel mutation strategy with a probability of 0.2 to refine the selected solution.
# The algorithm iterates through the population multiple times, each time evaluating the fitness of each individual and applying a mutation to the best individual in each iteration.
# The mutation strategy involves randomly mutating an individual with a probability of 0.2, which can lead to a better solution in some cases.
# The algorithm also evaluates the bounds of each individual and updates the fitness if the new individual is better.
# The process is repeated for a specified number of iterations, with a specified budget for each iteration.
# The algorithm uses a population size of 50 and a dimensionality of 10.
# The algorithm has not been tested on the BBOB test suite of 24 noiseless functions yet.
# ```python
# ```python