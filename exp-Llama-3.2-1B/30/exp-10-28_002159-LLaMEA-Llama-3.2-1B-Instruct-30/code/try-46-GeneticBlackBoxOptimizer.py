import random
import numpy as np

class GeneticBlackBoxOptimizer:
    def __init__(self, budget, dim, mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim
        self.mutation_rate = mutation_rate

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
        if random.random() < self.mutation_rate:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate(self, func):
        return func(np.array(self.population))

    def select(self, func, population, fitness_scores):
        # Select the fittest individuals
        fittest_individuals = population[np.argsort(fitness_scores)]
        # Select the next generation based on the probability of mutation
        next_generation = fittest_individuals[:self.population_size // 2]
        # Select the remaining individuals based on the probability of crossover
        remaining_individuals = fittest_individuals[self.population_size // 2:]
        # Combine the next generation and the remaining individuals
        new_population = np.concatenate((next_generation, remaining_individuals))
        return new_population

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        child = np.concatenate((parent1[:self.dim // 2], parent2[self.dim // 2:]))
        return child

    def __str__(self):
        return "Genetic Black Box Optimization"

# Description: Evolutionary Black Box Optimization using Genetic Algorithm with Mutation
# Code: 