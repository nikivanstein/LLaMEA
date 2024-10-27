import numpy as np
import random
import math

class EBA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.iterations = 0

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def mutate(individual):
            mutated_individual = individual.copy()
            if random.random() < 0.2:  # 20% chance of mutation
                mutated_individual[random.randint(0, self.dim-1)] += random.uniform(-5.0, 5.0)
            return mutated_individual

        def evaluate_fitness(individual):
            fitness = objective(individual)
            if fitness < self.fitnesses[individual, individual] + 1e-6:
                self.fitnesses[individual, individual] = fitness
                return individual
            else:
                return individual

        for _ in range(self.budget):
            for i in range(self.population_size):
                individual = evaluate_fitness(self.population[i])
                if individual is None:
                    individual = self.evaluate_fitness(random.choice(self.population))
                mutated_individual = mutate(individual)
                self.population[i] = mutated_individual

        return self.fitnesses

# One-line description with the main idea
# EBA: Evolutionary Algorithm for Black Box Optimization using Evolutionary Mutation
# 
# EBA combines the strengths of genetic algorithms and evolutionary mutation to optimize black box functions.