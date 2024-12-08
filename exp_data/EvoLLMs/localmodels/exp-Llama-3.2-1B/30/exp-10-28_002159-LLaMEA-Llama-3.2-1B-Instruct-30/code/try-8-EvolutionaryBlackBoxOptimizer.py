import random
import numpy as np
import copy

class EvolutionaryBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim
        self.algorithms = {}

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.search_spaces[i][0], self.search_spaces[i][1]) for i in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        def fitness(individual):
            return func(individual)

        def fitness_bounded(individual):
            return fitness(individual)

        for _ in range(self.budget):
            for i, individual in enumerate(self.population):
                fitness_scores[i] = fitness(individual)
            best_individual = self.population[np.argmax(fitness_scores)]
            new_individual = [random.uniform(self.search_spaces[j][0], self.search_spaces[j][1]) for j in range(self.dim)]
            new_individual = np.array(new_individual)
            if fitness(individual) > fitness(best_individual):
                self.population[i] = new_individual
                self.fitness_scores[i] = fitness(individual)

        return self.population

    def mutate(self, individual):
        if random.random() < 0.3:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate(self, func):
        return func(np.array(self.population))

    def adapt_algorithm(self, algorithm_name):
        if algorithm_name in self.algorithms:
            self.algorithms[algorithm_name].evaluate()
        else:
            algorithm = globals()[algorithm_name]
            self.algorithms[algorithm_name] = algorithm
            self.algorithms[algorithm_name].evaluate()
            if random.random() < 0.3:
                algorithm = copy.deepcopy(self.algorithms[algorithm_name])
            self.algorithms[algorithm_name] = algorithm

    def update(self):
        for algorithm in self.algorithms.values():
            algorithm.adapt_algorithm('EvolutionaryBlackBoxOptimizer')
        self.algorithms['EvolutionaryBlackBoxOptimizer'].evaluate()
        return self.algorithms['EvolutionaryBlackBoxOptimizer']

# Description: Evolutionary Black Box Optimization using Adaptive Line Search with Evolutionary Strategy
# Code: 