import numpy as np
import random
import copy

class SOLE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.population = self.initialize_population()
        self.best_individual = None
        self.best_func_value = float('inf')
        self.refine_probability = 0.15

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            individual = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(individual)
        return population

    def evaluate(self, func):
        if self.budget == 0:
            return

        for individual in self.population:
            func_value = func(*individual)
            if self.best_func_value > func_value:
                self.best_individual = copy.deepcopy(individual)
                self.best_func_value = func_value

        self.population = [self.mutate(individual) for individual in self.population]
        self.population = self.select_population()

    def mutate(self, individual):
        mutation_rate = 0.1
        mutated_individual = copy.deepcopy(individual)
        for i in range(self.dim):
            if random.random() < mutation_rate:
                mutated_individual[i] = random.uniform(-5.0, 5.0)
        return mutated_individual

    def select_population(self):
        population = []
        for _ in range(self.pop_size):
            individual = random.choice(self.population)
            population.append(individual)
        return population

    def refine_population(self):
        refined_population = []
        for individual in self.population:
            if random.random() < self.refine_probability:
                new_individual = individual
                for i in range(self.dim):
                    new_individual[i] = random.uniform(-5.0, 5.0)
                refined_population.append(new_individual)
            else:
                refined_population.append(individual)
        return refined_population

    def run(self, func):
        for _ in range(self.budget):
            if random.random() < 0.15:
                self.population = self.refine_population()
            self.evaluate(func)
        return self.best_individual, self.best_func_value

# Example usage:
def func(x):
    return np.sum(x**2)

sole = SOLE(budget=100, dim=10)
best_individual, best_func_value = sole.run(func)
print("Best individual:", best_individual)
print("Best function value:", best_func_value)