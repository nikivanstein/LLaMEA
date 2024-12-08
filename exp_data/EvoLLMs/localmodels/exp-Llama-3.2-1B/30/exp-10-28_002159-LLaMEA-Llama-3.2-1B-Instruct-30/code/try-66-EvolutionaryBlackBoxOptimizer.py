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
        self.algorithms = []

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
                self.algorithms.append((individual, best_individual, new_individual, self.fitness_scores[i]))

        return self.population

    def mutate(self, individual):
        if random.random() < 0.3:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate(self, func):
        return func(np.array(self.population))

    def select(self, algorithm):
        if len(self.algorithms) > 0:
            best_individual, best_fitness, new_individual, best_fitness_score = self.algorithms[0]
            best_individual = best_individual[np.argmax(best_fitness)]
            best_fitness = best_fitness[np.argmax(best_fitness)]
            new_individual = new_individual[np.argmax(new_fitness)]
            best_fitness_score = best_fitness_score[np.argmax(best_fitness_score)]
            return best_individual, best_fitness, new_individual, best_fitness_score
        else:
            return self.population[np.argmax(self.fitness_scores)]

    def mutate_population(self, algorithm):
        if len(self.algorithms) > 0:
            individual, best_individual, new_individual, best_fitness_score = algorithm
            new_individual = self.mutate(new_individual)
            individual = self.mutate(individual)
            algorithm = (individual, best_individual, new_individual, best_fitness_score)
            self.algorithms.append(algorithm)
        return algorithm

    def evolve(self, func):
        best_individual, best_fitness, new_individual, best_fitness_score = self.select(self.evaluate(func))
        new_individual = self.mutate_population((best_individual, best_fitness, new_individual, best_fitness_score))
        self.population = self.select(self.evaluate(func))
        return new_individual

# Description: Evolutionary Black Box Optimization Algorithm
# Code: 
# ```python
# import numpy as np
# import random
# import time
#
# def fitness(individual):
#     return np.sum(np.abs(individual - np.array([0, 0, 0, 1])))
#
# def mutate(individual):
#     if random.random() < 0.3:
#         index = random.randint(0, 3)
#         individual[index] += random.uniform(-1, 1)
#     return individual
#
# def select(algorithm):
#     return algorithm[0]
#
# def evolve(func):
#     start_time = time.time()
#     best_individual = select(func)
#     new_individual = mutate(best_individual)
#     end_time = time.time()
#     print(f"Mutation time: {end_time - start_time} seconds")
#     return new_individual
#
# optimizer = EvolutionaryBlackBoxOptimizer(1000, 4)
# best_individual = optimizer.evolve(fitness)
# print(f"Best individual: {best_individual}")
# print(f"Best fitness: {fitness(best_individual)}")
# print(f"Time taken: {optimizer.budget * 1000} milliseconds")