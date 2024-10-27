import numpy as np
import random

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.fitness = self.evaluate_fitness()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def evaluate_fitness(self):
        fitness = []
        for individual in self.population:
            fitness.append(self.func(individual))
        return fitness

    def __call__(self, func):
        for _ in range(self.budget):
            # Selection
            parents = self.select_parents()

            # Crossover
            offspring = self.crossover(parents)

            # Mutation
            offspring = self.mutate(offspring)

            # Replacement
            self.population = self.replace_population(offspring)

            # Refine strategy
            if random.random() < 0.3:
                self.refine_strategy()

    def select_parents(self):
        # Selection strategy
        parents = random.sample(self.population, self.population_size // 2)
        return parents

    def crossover(self, parents):
        offspring = []
        for _ in range(self.population_size // 2):
            parent1, parent2 = random.sample(parents, 2)
            child = (parent1 + parent2) / 2
            offspring.append(child)
        return offspring

    def mutate(self, offspring):
        # Mutation strategy
        for i in range(self.population_size // 2):
            if random.random() < 0.1:
                mutation = np.random.uniform(-1.0, 1.0)
                offspring[i] += mutation
        return offspring

    def replace_population(self, offspring):
        self.population = offspring
        self.fitness = self.evaluate_fitness()
        return self.fitness

    def refine_strategy(self):
        # Refine strategy
        for i in range(self.population_size):
            if random.random() < 0.3:
                self.population[i] += np.random.uniform(-1.0, 1.0)

    def func(self, x):
        # Black box function
        return np.sum(x**2)

# Example usage
budget = 100
dim = 10
algorithm = HybridEvolutionaryAlgorithm(budget, dim)
algorithm()