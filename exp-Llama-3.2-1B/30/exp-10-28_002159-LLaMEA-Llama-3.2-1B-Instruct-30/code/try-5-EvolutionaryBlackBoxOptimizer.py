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
        self.mutation_rate = 0.01
        self.crossover_rate = 0.5

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

            # Adapt the mutation rate based on the fitness
            if fitness(individual) > 0.5:
                self.mutation_rate = 0.1

            # Perform crossover between the best individual and the new individual
            if random.random() < 0.5:
                parent1 = self.population[np.argmax(fitness_scores)]
                parent2 = self.population[np.random.randint(0, self.population_size)]
                child = (parent1 + parent2) / 2
                self.population[np.argmax(fitness_scores)] = child
                self.fitness_scores[np.argmax(fitness_scores)] = fitness(child)

        return self.population

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            index = random.randint(0, self.dim - 1)
            self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
        return individual

    def evaluate(self, func):
        return func(np.array(self.population))

# One-line description: Evolutionary Black Box Optimization with adaptive mutation and crossover to improve the search space exploration and exploitation.
# Code: 