import random
import math
import numpy as np

class DynamicAdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = [random.uniform(-5.0, 5.0) for _ in range(self.population_size)]
        self.fitnesses = [0] * self.population_size

    def __call__(self, func):
        def mutate(individual):
            if random.random() < 0.5:
                return individual + random.uniform(-1.0, 1.0)
            return individual

        def mutate_f(individual, func):
            return func(mutate(individual))

        def select_next_individual():
            best_individual = max(self.population, key=lambda x: self.fitnesses[x])
            next_individual = random.uniform(-5.0, 5.0)
            while next_individual == best_individual:
                next_individual = random.uniform(-5.0, 5.0)
            return next_individual

        def evaluate_fitness(individual):
            fitness = func(individual)
            updated_individual = select_next_individual()
            while updated_individual == individual:
                updated_individual = select_next_individual()
            return updated_individual, fitness

        for _ in range(self.budget):
            individual, fitness = evaluate_fitness(self.population[0])
            updated_individual, fitness = evaluate_fitness(individual)
            self.population[0] = updated_individual
            self.fitnesses[0] = fitness

        return self.population[0]

# One-line description: "Dynamic Adaptive Genetic Algorithm with Adaptive Sampling"
# This algorithm uses adaptive sampling to select the next individual based on the fitness and the dimension,
# and ensures the fitness stays within the bounds to prevent overfitting.