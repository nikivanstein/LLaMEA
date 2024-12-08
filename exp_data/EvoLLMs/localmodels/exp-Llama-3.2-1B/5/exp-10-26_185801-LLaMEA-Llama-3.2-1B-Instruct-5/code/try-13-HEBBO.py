import numpy as np
import random
import operator

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        return [random.uniform(-5.0, 5.0) for _ in range(self.dim)]

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            new_individual = self.evaluate_fitness(self.population)
            if np.isnan(new_individual) or np.isinf(new_individual):
                raise ValueError("Invalid function value")
            if new_individual < 0 or new_individual > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.population = new_individual
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return self.population

    def evaluate_fitness(self, individual):
        func_value = individual
        for _ in range(self.dim):
            func_value = func_value * (self.search_space + 1)
        return func_value

    def mutate(self, individual):
        if random.random() < 0.05:
            idx = random.randint(0, self.dim - 1)
            self.search_space[idx] += random.uniform(-0.1, 0.1)
        return individual

    def crossover(self, parent1, parent2):
        if random.random() < 0.5:
            return np.concatenate((parent1[:random.randint(0, len(parent1) - 1)], parent2[random.randint(0, len(parent2) - 1)]))
        else:
            return parent1

# One-line description with main idea:
# Evolutionary Algorithm for Black Box Optimization (EA-BBO)
# Uses mutation and crossover operators to refine the solution with a probability of 0.05