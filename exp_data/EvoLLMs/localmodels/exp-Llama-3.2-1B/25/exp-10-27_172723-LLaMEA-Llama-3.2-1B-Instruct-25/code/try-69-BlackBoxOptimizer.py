import random
import numpy as np
import math

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = []

    def __call__(self, func):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

    def mutate(self, individual):
        # Randomly change two genes in the individual
        idx1, idx2 = random.sample(range(self.dim), 2)
        individual[idx1], individual[idx2] = random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)

    def crossover(self, parent1, parent2):
        # Select two parents and create a child by combining them
        child = np.concatenate((parent1[:self.dim//2], parent2[self.dim//2:]))
        return child

    def selection(self):
        # Select the fittest individuals
        fitness = np.array([func(individual) for individual in self.population])
        sorted_indices = np.argsort(fitness)
        return [self.population[i] for i in sorted_indices[:self.budget//2]]

    def evolve(self, population_size, mutation_rate, crossover_rate):
        # Evolve the population using the selection, crossover, and mutation operators
        for _ in range(self.budget):
            population = self.selection()
            for individual in population:
                self.mutate(individual)
                if random.random() < mutation_rate:
                    self.mutate(individual)
            population = self.crossover(population, population)
            population = self.selection()
            if random.random() < crossover_rate:
                self.population = population

# One-line description: "Meta-Heuristic Optimization for Black Box Functions"
# Code: 