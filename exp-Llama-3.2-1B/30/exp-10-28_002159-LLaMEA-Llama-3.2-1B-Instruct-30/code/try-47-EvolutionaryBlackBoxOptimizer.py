# Description: Evolutionary Black Box Optimization using Evolutionary Neighborhood Search
# Code: 
# ```python
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
        self.neighborhood_size = 2  # Change this to adjust neighborhood size

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.search_spaces[i][0], self.search_spaces[i][1]) for i in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        def fitness(individual):
            return func(individual)

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

    def select_neighbor(self, individual):
        return self.population[np.random.randint(0, self.population_size)]

    def __next_generation(self, individual):
        neighbors = [self.select_neighbor(individual) for _ in range(self.neighborhood_size)]
        new_individual = [random.uniform(self.search_spaces[j][0], self.search_spaces[j][1]) for j in range(self.dim)]
        new_individual = np.array(new_individual)
        for i, neighbor in enumerate(neighbors):
            new_individual[i] = random.uniform(self.search_spaces[j][0], self.search_spaces[j][1]) for j in range(self.dim)
        return new_individual

    def next_generation(self):
        new_individual = self.__next_generation(self.population[np.argmax(self.fitness_scores)])
        self.population[np.argmax(self.fitness_scores)] = new_individual
        self.fitness_scores[np.argmax(self.fitness_scores)] = self.evaluate(new_individual)
        return new_individual

    def mutate_next_generation(self):
        for i, individual in enumerate(self.population):
            if random.random() < 0.3:
                self.search_spaces[i] = (self.search_spaces[i][0] + random.uniform(-1, 1), self.search_spaces[i][1] + random.uniform(-1, 1))
        return self.population