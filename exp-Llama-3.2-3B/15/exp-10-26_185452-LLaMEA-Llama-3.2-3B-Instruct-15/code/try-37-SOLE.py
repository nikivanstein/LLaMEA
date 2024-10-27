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
        self.mutation_rate = 0.1
        self.local_search_rate = 0.15

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
        self.local_search()

    def mutate(self, individual):
        mutated_individual = copy.deepcopy(individual)
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                mutated_individual[i] = random.uniform(-5.0, 5.0)
        return mutated_individual

    def select_population(self):
        population = []
        for _ in range(self.pop_size):
            individual = random.choice(self.population)
            population.append(individual)
        return population

    def local_search(self):
        for individual in self.population:
            local_search_rate = self.local_search_rate
            if random.random() < local_search_rate:
                improved = False
                for i in range(self.dim):
                    new_individual = copy.deepcopy(individual)
                    new_individual[i] += 0.1
                    func_value = func(*new_individual)
                    if func_value < func(*individual):
                        improved = True
                        individual[i] = new_individual[i]
                if improved:
                    self.population = [self.mutate(individual) for individual in self.population]
                    self.population = self.select_population()

    def run(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
        return self.best_individual, self.best_func_value

# Example usage:
def func(x):
    return np.sum(x**2)

sole = SOLE(budget=100, dim=10)
best_individual, best_func_value = sole.run(func)
print("Best individual:", best_individual)
print("Best function value:", best_func_value)