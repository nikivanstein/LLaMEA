import random
import numpy as np

class EvolutionaryBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim
        self.refine_strategy = False

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.search_spaces[i][0], self.search_spaces[i][1]) for i in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        def fitness(individual):
            return func(individual)

        def mutate(individual):
            if random.random() < 0.01:
                index = random.randint(0, self.dim - 1)
                self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
            return individual

        for _ in range(self.budget):
            for i, individual in enumerate(self.population):
                fitness_scores[i] = fitness(individual)
                best_individual = self.population[np.argmax(fitness_scores)]
                if not self.refine_strategy and fitness(individual) > fitness(best_individual):
                    self.population[i] = mutate(individual)
                    self.fitness_scores[i] = fitness(individual)
                elif random.random() < 0.3:
                    self.population[i] = mutate(individual)

        return self.population

    def evaluate(self, func):
        return func(np.array(self.population))

# One-line description with main idea
# Evolutionary Black Box Optimization with Refining Strategy
# 
# This algorithm optimizes the black box function using evolutionary algorithms with a refining strategy.