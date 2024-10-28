# Description: "Black Box Optimization using Genetic Algorithms"
# Code: 
# ```python
import random
import numpy as np
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]  # Define the search space
        self.population_size = 100  # Population size
        self.mutation_rate = 0.01  # Mutation rate
        self.crossover_rate = 0.5  # Crossover rate
        self.population = self.generate_population()  # Generate initial population

    def generate_population(self):
        population = []
        for _ in range(self.population_size):
            individual = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            population.append(individual)
        return population

    def evaluate_fitness(self, individual, func):
        fitness = func(individual)
        return fitness

    def select_parents(self, population):
        parents = []
        for _ in range(self.population_size // 2):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            while parent1 == parent2:
                parent2 = random.choice(population)
            parent1_fitness = self.evaluate_fitness(parent1, func)
            parent2_fitness = self.evaluate_fitness(parent2, func)
            if parent1_fitness > parent2_fitness:
                parents.append(parent1)
            else:
                parents.append(parent2)
        return parents

    def crossover(self, parents):
        offspring = []
        for _ in range(self.population_size // 2):
            parent1, parent2 = random.sample(parents, 2)
            child1 = (parent1[0] + parent2[0]) / 2, (parent1[1] + parent2[1]) / 2
            offspring.append(child1)
        return offspring

    def mutate(self, offspring):
        for individual in offspring:
            if random.random() < self.mutation_rate:
                individual = (individual[0] + random.uniform(-1, 1), individual[1] + random.uniform(-1, 1))
        return offspring

    def mutate_offspring(self, offspring):
        for individual in offspring:
            if random.random() < self.mutation_rate:
                individual = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
        return offspring

    def evaluate_bboh(self, func, population):
        offspring = self.mutate_offspring(population)
        results = []
        for individual in population:
            fitness = self.evaluate_fitness(individual, func)
            results.append((individual, fitness))
        return results

    def select_parents(self, results):
        parents = []
        for individual, fitness in results:
            parent_fitness = self.evaluate_fitness(individual, func)
            if parent_fitness < fitness:
                parents.append(individual)
        return parents

    def crossover_and_mutate(self, parents):
        offspring = self.crossover(parents)
        offspring = self.mutate_offspring(offspring)
        return offspring

    def run(self, func, population_size):
        for _ in range(1000):
            population = self.generate_population()
            results = self.evaluate_bboh(func, population)
            parents = self.select_parents(results)
            offspring = self.crossover_and_mutate(parents)
            self.population = offspring
            if self.evaluate_bboh(func, self.population)[-1][1] < self.budget:
                break
        return self.population

# Define a noiseless function
def noiseless_func(x):
    return np.sin(x)

# Define a noise function
def noise_func(x):
    return np.random.normal(0, 1, x)

# Define a test function
def test_func(x):
    return x**2 + 2*x + 1

# Create an instance of the GeneticAlgorithm class
ga = GeneticAlgorithm(100, 10)

# Optimize the test function using the GeneticAlgorithm
best_func = ga.run(test_func, 100)

# Print the best function found
print("Best function:", best_func)
print("Best fitness:", best_func.budget)