import numpy as np
import random

class PEAHM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.memory_size = 10
        self.population = self.initialize_population()
        self.memory = self.initialize_memory()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def initialize_memory(self):
        memory = []
        for _ in range(self.memory_size):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            memory.append(individual)
        return memory

    def evaluate(self, func):
        for individual in self.population:
            func_value = func(individual)
            self.population.append((individual, func_value))

        self.population = sorted(self.population, key=lambda x: x[1])

        if len(self.population) > self.budget:
            self.population = self.population[:self.budget]

    def update_memory(self):
        self.memory = []
        for individual in self.population:
            if individual not in self.memory:
                self.memory.append(individual)

    def update_population(self):
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(self.population, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        self.population = new_population

    def crossover(self, parent1, parent2):
        child = np.zeros(self.dim)
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child

    def mutate(self, individual):
        mutation_rate = 0.1
        for i in range(self.dim):
            if random.random() < mutation_rate:
                individual[i] += np.random.uniform(-1.0, 1.0)
        return individual

    def update(self, func):
        self.evaluate(func)
        self.update_memory()
        self.update_population()

        # Refine the strategy of the best individual with probability 0.3
        if random.random() < 0.3:
            best_individual, _ = self.population[0]
            best_individual = np.array(best_individual)
            best_individual = best_individual * 0.9 + np.random.uniform(-0.1, 0.1, self.dim)
            self.population[0] = (best_individual, self.population[0][1])

# Example usage:
def func(x):
    return np.sum(x**2)

peahm = PEAHM(budget=10, dim=5)
peahm.update(func)