import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.01
        self.population = self.generate_population()

    def generate_population(self):
        return [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

    def evaluate_fitness(self, individual):
        func = lambda x: x**2 + 0.5*x + 0.1
        return func(individual)

    def __call__(self, func, bounds):
        new_population = self.population.copy()
        for _ in range(self.budget):
            for i in range(self.population_size):
                if random.random() < 0.5:
                    new_individual = func(new_population[i])
                    if random.random() < 0.2:
                        new_individual = random.uniform(bounds[0], bounds[1])
                    if random.random() < 0.4:
                        new_individual = random.uniform(bounds[0], bounds[1])
                    new_population[i] = new_individual
        return new_population

    def fitness(self, individual):
        return self.evaluate_fitness(individual)

    def selection(self, fitness):
        return [individual for _, individual in sorted(zip(fitness, self.population), reverse=True)]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.dim)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutation(self, individual):
        if random.random() < self.mutation_rate:
            idx = random.randint(0, self.dim-1)
            individual[idx] = random.uniform(-5.0, 5.0)
        return individual

    def run(self):
        fitness = [self.fitness(individual) for individual in self.population]
        best_individual = min(fitness, key=fitness.index)
        best_fitness = fitness[fitness.index(best_individual)]
        print("Best individual:", best_individual)
        print("Best fitness:", best_fitness)
        return best_individual, best_fitness

# Description: Genetic Algorithm with Randomized Mutation
# Code: 