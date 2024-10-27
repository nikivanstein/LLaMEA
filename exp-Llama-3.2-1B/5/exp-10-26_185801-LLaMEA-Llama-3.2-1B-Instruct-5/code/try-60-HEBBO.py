import numpy as np
from scipy.optimize import minimize

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

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

    def select_solution(self, current_solution, current_fitness, current_individual):
        # Select a parent based on the fitness
        parent1 = current_solution
        parent2 = current_individual
        if np.random.rand() < 0.5:
            parent1 = current_individual
        else:
            parent2 = current_solution

        # Select a child based on the parent selection strategy
        if np.random.rand() < 0.5:
            child = parent1 + np.random.normal(0, 1, self.dim)
        else:
            child = parent2 - np.random.normal(0, 1, self.dim)

        # Evaluate the child
        child_fitness = self.func_evaluations * current_fitness + self.budget * self.func_evaluations + np.random.normal(0, 0.1, self.dim)
        if child_fitness < current_fitness:
            child = current_solution

        return child, child_fitness

    def mutate(self, individual):
        # Randomly mutate the individual
        if np.random.rand() < 0.5:
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        else:
            self.search_space = np.linspace(5.0, 5.0, self.dim)
        return individual

    def crossover(self, parent1, parent2):
        # Perform crossover
        child = np.concatenate((parent1, parent2))
        return child

    def evolve(self, population_size):
        # Evolve the population
        population = [self.select_solution(np.random.choice(self.search_space, size=self.dim), np.random.rand(), np.random.choice(self.search_space, size=self.dim)) for _ in range(population_size)]
        return population

    def optimize(self, func, population_size):
        # Evolve the population using HEBBO
        population = self.evolve(population_size)
        return self.__call__(func)

# Description: HEBBO - Hybrid Evolutionary Bounded Optimization using Binary and Continuous Optimization
# Code: 