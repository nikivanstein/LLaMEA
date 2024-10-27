# Description: Novel metaheuristic algorithm to optimize black box functions using evolutionary strategies.
# Code: 
import numpy as np
import random

class EvolutionaryOptimization:
    def __init__(self, budget, dim, mutation_rate):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = mutation_rate
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            if random.random() < self.mutation_rate:
                idx = random.randint(0, self.dim - 1)
                individual[idx] = (individual[idx] + 1) / 2
            return individual

        def evaluate_fitness(individual):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual] + 1e-6:
                self.fitnesses[individual] = fitness
                self.population[individual] = individual

        for _ in range(self.budget):
            for i in range(self.population_size):
                fitness = objective(self.population[i])
                if fitness < self.fitnesses[i] + 1e-6:
                    self.fitnesses[i] = fitness
                    self.population[i] = mutate(self.population[i])

        # Refine the solution by changing the individual lines of the selected solution
        # to refine its strategy
        for i in range(self.population_size):
            if random.random() < 0.2:
                self.population[i] = mutate(self.population[i])

        # Store the updated population history
        self.population_history.append(self.population)

        return self.fitnesses

# One-line description with the main idea
# Novel metaheuristic algorithm to optimize black box functions using evolutionary strategies.
# 
# The algorithm uses a population of individuals, each representing a possible solution,
# and evolves the population through a series of iterations, refining the solution
# by changing individual lines of the selected solution to refine its strategy.
# 
# The mutation rate determines the frequency of individual line changes, with higher
# rates leading to more aggressive exploration and higher rates leading to more
# conservative exploration.