import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        return [[random.uniform(-5.0, 5.0) for _ in range(self.dim)] for _ in range(self.population_size)]

    def __call__(self, func):
        for _ in range(self.budget):
            for individual in self.population:
                func(individual)
                if random.random() < 0.5:  # Refine strategy with probability 0.45
                    self.population[_] = [x for x in individual if func(x) < func(self.population[_])]
        return self.population

    def select_solution(self):
        return random.choice(self.population)

    def evaluate_function(self, func, individual):
        return func(individual)

    def fitness(self, func, individual):
        return -self.evaluate_function(func, individual)

    def mutate(self, individual):
        mutated_individual = [x + random.uniform(-0.1, 0.1) for x in individual]
        return mutated_individual

    def crossover(self, parent1, parent2):
        child = [x for x in parent1 if x not in parent2]
        for x in parent2:
            if random.random() < 0.5:  # Refine strategy with probability 0.45
                child.append(x)
        return child

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 