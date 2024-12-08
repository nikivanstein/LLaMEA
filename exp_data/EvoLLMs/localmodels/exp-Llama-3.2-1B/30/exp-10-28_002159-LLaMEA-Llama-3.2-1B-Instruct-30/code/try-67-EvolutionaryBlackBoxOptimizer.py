# Description: Evolutionary Black Box Optimization using Evolutionary Algorithm with Adaptive Mutation
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
        self mutation_rate = 0.01
        self.best_individual = None
        self.best_fitness = -np.inf

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
                if fitness_scores[i] > self.best_fitness:
                    self.best_individual = individual
                    self.best_fitness = fitness_scores[i]

            best_individual = self.best_individual
            for i, individual in enumerate(self.population):
                fitness_scores[i] = fitness(individual)
                if fitness_scores[i] > self.best_fitness:
                    self.best_individual = individual
                    self.best_fitness = fitness_scores[i]

            new_individual = [random.uniform(self.search_spaces[j][0], self.search_spaces[j][1]) for j in range(self.dim)]
            new_individual = np.array(new_individual)
            if random.random() < self.mutation_rate:
                index = random.randint(0, self.dim - 1)
                self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))

            self.population[i] = new_individual
            self.fitness_scores[i] = fitness(individual)

        return self.population

    def mutate(self, individual):
        if random.random() < 0.3:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))

        return individual

# Description: Evolutionary Black Box Optimization using Evolutionary Algorithm with Adaptive Mutation
# Code: 
# ```python
evbbo = EvolutionaryBlackBoxOptimizer(budget=1000, dim=5)
print("Best individual:", evbbo.population[np.argmax(evbbo.fitness_scores)])
print("Best fitness:", evbbo.best_fitness)