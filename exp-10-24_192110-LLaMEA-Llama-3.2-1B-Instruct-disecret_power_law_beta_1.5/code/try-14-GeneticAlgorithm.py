import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        return [(np.random.uniform(-5.0, 5.0, self.dim), np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(self.population_size)]

    def fitness(self, func, func_evals):
        func_evals = np.array(func_evals)
        return -np.sum(func_evals)

    def __call__(self, func):
        func_evals = np.random.randint(0, self.budget, self.dim**2)
        fitness = self.fitness(func, func_evals)
        while fitness!= -np.inf:
            idx = np.random.randint(0, self.dim**2)
            new_func = func
            new_func_evals = func_evals.copy()
            new_func_evals[idx] = func_evals[idx] + random.uniform(-1, 1)
            new_fitness = self.fitness(new_func, new_func_evals)
            if new_fitness < fitness:
                func_evals = new_func_evals
                fitness = new_fitness
            else:
                func_evals = np.append(func_evals, new_func_evals[idx])
                fitness = -np.inf
        return func, func_evals

    def select(self, func, func_evals):
        probabilities = np.array([func_evals[i] / np.sum(func_evals) for i in range(len(func_evals))])
        selected_idx = np.random.choice(len(func_evals), size=self.population_size, p=probabilities)
        return selected_idx, probabilities

    def crossover(self, parent1, parent2):
        idx1 = np.random.randint(0, self.dim**2)
        idx2 = np.random.randint(0, self.dim**2)
        child1 = parent1.copy()
        child1[idx1] = parent2[idx1]
        child2 = parent2.copy()
        child2[idx2] = parent1[idx2]
        return child1, child2

    def mutate(self, func, func_evals):
        idx = np.random.randint(0, self.dim**2)
        new_func_evals = func_evals.copy()
        new_func_evals[idx] = func_evals[idx] + random.uniform(-1, 1)
        return func, new_func_evals