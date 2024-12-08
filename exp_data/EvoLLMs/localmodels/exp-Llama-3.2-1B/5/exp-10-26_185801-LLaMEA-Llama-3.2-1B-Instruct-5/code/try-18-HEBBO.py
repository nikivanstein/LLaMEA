import numpy as np
import random

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = []

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def select_solution(self):
        if len(self.population) == 0:
            return self.__call__(self.__init__)
        else:
            return random.choice(self.population)

    def mutate(self, individual):
        if len(individual) == 0:
            return individual
        new_individual = individual.copy()
        if random.random() < 0.05:
            idx = random.randint(0, self.dim-1)
            new_individual[idx] += random.uniform(-1, 1)
        return new_individual

    def crossover(self, parent1, parent2):
        if len(parent1) == 0 or len(parent2) == 0:
            return parent1, parent2
        child1 = parent1.copy()
        child2 = parent2.copy()
        if random.random() < 0.5:
            child1, child2 = self.mutate(child1), self.mutate(child2)
        return child1, child2

    def evaluate_fitness(self, individual):
        func_value = self.__call__(individual)
        if np.isnan(func_value) or np.isinf(func_value):
            raise ValueError("Invalid function value")
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        return func_value

    def run(self, func):
        population = [self.select_solution() for _ in range(100)]
        for _ in range(100):
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = self.crossover(parent1, parent2)
            population.append(self.evaluate_fitness(child1))
            population.append(self.evaluate_fitness(child2))
        return population

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 