import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        return [np.random.uniform(self.search_space[0], self.search_space[1], self.dim) for _ in range(self.population_size)]

    def fitness(self, individual):
        func_value = self.evaluate_function(individual)
        return func_value

    def evaluate_function(self, individual):
        func = lambda x: individual
        func_value = func(self.search_space)
        if np.isnan(func_value) or np.isinf(func_value):
            raise ValueError("Invalid function value")
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        return func_value

    def selection(self):
        return self.population[np.random.choice(len(self.population), self.population_size, replace=False)]

    def crossover(self, parent1, parent2):
        child = np.zeros(self.dim)
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child

    def mutation(self, individual):
        if random.random() < 0.05:
            index = random.randint(0, self.dim - 1)
            individual[index] = np.random.uniform(self.search_space[index])
        return individual

    def next_generation(self):
        offspring = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(self.population, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutation(child)
            offspring.append(child)
        return offspring

    def run(self, iterations):
        for _ in range(iterations):
            self.population = self.next_generation()
            best_individual = max(self.population, key=self.fitness)
            if self.f(best_individual) > self.f(self.population[0]):
                self.population[0] = best_individual
        return self.population[0]

    def run_bbb(self, func):
        # Run the BBOB test suite
        results = {}
        for func_name in ['sphere', 'cube', 'box', 'ring', 'cylindrical', 'elliptical']:
            func_value = func(self.search_space)
            results[func_name] = func_value
        return results

# Description: Genetic Algorithm for Black Box Optimization
# Code: 