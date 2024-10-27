import numpy as np
import random
import string

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.random_search_rate = 0.2

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        # Initialize population
        population = [np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim) for _ in range(self.population_size)]

        # Evaluate population
        fitnesses = []
        for individual in population:
            fitnesses.append(func(individual))

        # Select parents
        parents = []
        for _ in range(self.population_size):
            parent_index = np.random.choice(self.population_size)
            parents.append(population[parent_index])

        # Crossover
        offspring = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(parents, 2)
            child = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
            for i in range(self.dim):
                if random.random() < self.crossover_rate:
                    child[i] = (parent1[i] + parent2[i]) / 2
            offspring.append(child)

        # Mutation
        for individual in offspring:
            if random.random() < self.mutation_rate:
                mutation = np.random.uniform(-1, 1, self.dim)
                individual += mutation

        # Random search
        for _ in range(int(self.budget * self.random_search_rate)):
            new_individual = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
            fitness = func(new_individual)
            if fitness < np.min(fitnesses):
                population[fitnesses.index(np.min(fitnesses))] = new_individual
                fitnesses.append(fitness)

        # Evaluate new population
        new_fitnesses = []
        for individual in population:
            new_fitnesses.append(func(individual))

        # Replace worst individual
        worst_index = np.argmin(new_fitnesses)
        population[worst_index] = offspring[np.random.choice(len(offspring))]

        # Return best individual
        return np.min(population), population[np.argmin(new_fitnesses)]

# Usage
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1])

def f3(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1])

def f4(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1])

def f5(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1])

def f6(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1])

def f7(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1])

def f8(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1])

def f9(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1]) + 0.00000001 * np.sin(100000000 * x[0]) + 0.00000001 * np.sin(100000000 * x[1])

def f10(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1]) + 0.00000001 * np.sin(100000000 * x[0]) + 0.00000001 * np.sin(100000000 * x[1]) + 0.000000001 * np.sin(1000000000 * x[0]) + 0.000000001 * np.sin(1000000000 * x[1])

def f11(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1]) + 0.00000001 * np.sin(100000000 * x[0]) + 0.00000001 * np.sin(100000000 * x[1]) + 0.000000001 * np.sin(1000000000 * x[0]) + 0.000000001 * np.sin(1000000000 * x[1]) + 0.0000000001 * np.sin(10000000000 * x[0]) + 0.0000000001 * np.sin(10000000000 * x[1])

def f12(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1]) + 0.00000001 * np.sin(100000000 * x[0]) + 0.00000001 * np.sin(100000000 * x[1]) + 0.000000001 * np.sin(1000000000 * x[0]) + 0.000000001 * np.sin(1000000000 * x[1]) + 0.0000000001 * np.sin(10000000000 * x[0]) + 0.0000000001 * np.sin(10000000000 * x[1]) + 0.00000000001 * np.sin(100000000000 * x[0]) + 0.00000000001 * np.sin(100000000000 * x[1])

def f13(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1]) + 0.00000001 * np.sin(100000000 * x[0]) + 0.00000001 * np.sin(100000000 * x[1]) + 0.000000001 * np.sin(1000000000 * x[0]) + 0.000000001 * np.sin(1000000000 * x[1]) + 0.0000000001 * np.sin(10000000000 * x[0]) + 0.0000000001 * np.sin(10000000000 * x[1]) + 0.00000000001 * np.sin(100000000000 * x[0]) + 0.00000000001 * np.sin(100000000000 * x[1])

def f14(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1]) + 0.00000001 * np.sin(100000000 * x[0]) + 0.00000001 * np.sin(100000000 * x[1]) + 0.000000001 * np.sin(1000000000 * x[0]) + 0.000000001 * np.sin(1000000000 * x[1]) + 0.0000000001 * np.sin(10000000000 * x[0]) + 0.0000000001 * np.sin(10000000000 * x[1]) + 0.00000000001 * np.sin(100000000000 * x[0]) + 0.00000000001 * np.sin(100000000000 * x[1])

def f15(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1]) + 0.00000001 * np.sin(100000000 * x[0]) + 0.00000001 * np.sin(100000000 * x[1]) + 0.000000001 * np.sin(1000000000 * x[0]) + 0.000000001 * np.sin(1000000000 * x[1]) + 0.0000000001 * np.sin(10000000000 * x[0]) + 0.0000000001 * np.sin(10000000000 * x[1]) + 0.00000000001 * np.sin(100000000000 * x[0]) + 0.00000000001 * np.sin(100000000000 * x[1])

def f16(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1]) + 0.00000001 * np.sin(100000000 * x[0]) + 0.00000001 * np.sin(100000000 * x[1]) + 0.000000001 * np.sin(1000000000 * x[0]) + 0.000000001 * np.sin(1000000000 * x[1]) + 0.0000000001 * np.sin(10000000000 * x[0]) + 0.0000000001 * np.sin(10000000000 * x[1]) + 0.00000000001 * np.sin(100000000000 * x[0]) + 0.00000000001 * np.sin(100000000000 * x[1])

def f17(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1]) + 0.00000001 * np.sin(100000000 * x[0]) + 0.00000001 * np.sin(100000000 * x[1]) + 0.000000001 * np.sin(1000000000 * x[0]) + 0.000000001 * np.sin(1000000000 * x[1]) + 0.0000000001 * np.sin(10000000000 * x[0]) + 0.0000000001 * np.sin(10000000000 * x[1]) + 0.00000000001 * np.sin(100000000000 * x[0]) + 0.00000000001 * np.sin(100000000000 * x[1])

def f18(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1]) + 0.00000001 * np.sin(100000000 * x[0]) + 0.00000001 * np.sin(100000000 * x[1]) + 0.000000001 * np.sin(1000000000 * x[0]) + 0.000000001 * np.sin(1000000000 * x[1]) + 0.0000000001 * np.sin(10000000000 * x[0]) + 0.0000000001 * np.sin(10000000000 * x[1]) + 0.00000000001 * np.sin(100000000000 * x[0]) + 0.00000000001 * np.sin(100000000000 * x[1])

def f19(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1]) + 0.00000001 * np.sin(100000000 * x[0]) + 0.00000001 * np.sin(100000000 * x[1]) + 0.000000001 * np.sin(1000000000 * x[0]) + 0.000000001 * np.sin(1000000000 * x[1]) + 0.0000000001 * np.sin(10000000000 * x[0]) + 0.0000000001 * np.sin(10000000000 * x[1]) + 0.00000000001 * np.sin(100000000000 * x[0]) + 0.00000000001 * np.sin(100000000000 * x[1])

def f20(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1]) + 0.00000001 * np.sin(100000000 * x[0]) + 0.00000001 * np.sin(100000000 * x[1]) + 0.000000001 * np.sin(1000000000 * x[0]) + 0.000000001 * np.sin(1000000000 * x[1]) + 0.0000000001 * np.sin(10000000000 * x[0]) + 0.0000000001 * np.sin(10000000000 * x[1]) + 0.00000000001 * np.sin(100000000000 * x[0]) + 0.00000000001 * np.sin(100000000000 * x[1])

def f21(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1]) + 0.00000001 * np.sin(100000000 * x[0]) + 0.00000001 * np.sin(100000000 * x[1]) + 0.000000001 * np.sin(1000000000 * x[0]) + 0.000000001 * np.sin(1000000000 * x[1]) + 0.0000000001 * np.sin(10000000000 * x[0]) + 0.0000000001 * np.sin(10000000000 * x[1]) + 0.00000000001 * np.sin(100000000000 * x[0]) + 0.00000000001 * np.sin(100000000000 * x[1])

def f22(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1]) + 0.00000001 * np.sin(100000000 * x[0]) + 0.00000001 * np.sin(100000000 * x[1]) + 0.000000001 * np.sin(1000000000 * x[0]) + 0.000000001 * np.sin(1000000000 * x[1]) + 0.0000000001 * np.sin(10000000000 * x[0]) + 0.0000000001 * np.sin(10000000000 * x[1]) + 0.00000000001 * np.sin(100000000000 * x[0]) + 0.00000000001 * np.sin(100000000000 * x[1])

def f23(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1]) + 0.00000001 * np.sin(100000000 * x[0]) + 0.00000001 * np.sin(100000000 * x[1]) + 0.000000001 * np.sin(1000000000 * x[0]) + 0.000000001 * np.sin(1000000000 * x[1]) + 0.0000000001 * np.sin(10000000000 * x[0]) + 0.0000000001 * np.sin(10000000000 * x[1]) + 0.00000000001 * np.sin(100000000000 * x[0]) + 0.00000000001 * np.sin(100000000000 * x[1])

def f24(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.sin(10 * x[1]) + 0.01 * np.sin(100 * x[0]) + 0.01 * np.sin(100 * x[1]) + 0.001 * np.sin(1000 * x[0]) + 0.001 * np.sin(1000 * x[1]) + 0.0001 * np.sin(10000 * x[0]) + 0.0001 * np.sin(10000 * x[1]) + 0.00001 * np.sin(100000 * x[0]) + 0.00001 * np.sin(100000 * x[1]) + 0.000001 * np.sin(1000000 * x[0]) + 0.000001 * np.sin(1000000 * x[1]) + 0.0000001 * np.sin(10000000 * x[0]) + 0.0000001 * np.sin(10000000 * x[1]) + 0.00000001 * np.sin(100000000 * x[0]) + 0.00000001 * np.sin(100000000 * x[1]) + 0.000000001 * np.sin(1000000000 * x[0]) + 0.000000001 * np.sin(1000000000 * x[1]) + 0.0000000001 * np.sin(10000000000 * x[0]) + 0.0000000001 * np.sin(10000000000 * x[1]) + 0.00000000001 * np.sin(100000000000 * x[0]) + 0.00000000001 * np.sin(100000000000 * x[1])

hybrid_evolutionary_algorithm = HybridEvolutionaryAlgorithm(budget=10, dim=2)

def evaluate_bbob(func, algorithm):
    # Initialize logger
    logger = {}
    logger['name'] = 'HybridEvolutionaryAlgorithm'
    logger['description'] = 'Hybrid Evolutionary Algorithm with Crossover, Mutation, and Random Search for Black Box Optimization'
    logger['score'] = -np.inf

    # Evaluate function
    best_fitness = np.inf
    for _ in range(algorithm.budget):
        fitness, individual = algorithm()
        if fitness < best_fitness:
            best_fitness = fitness
            logger['best_individual'] = individual
            logger['best_fitness'] = fitness

    # Update score
    algorithm.score = best_fitness

    return logger

# Usage
logger = evaluate_bbob(f1, hybrid_evolutionary_algorithm)
print(logger)