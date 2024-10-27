import random
import numpy as np
import copy
from collections import deque
import math

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

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
        # Randomly swap two elements in the individual
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
        return individual

    def crossover(self, parent1, parent2):
        # Randomly select a crossover point
        crossover_point = random.randint(1, len(parent1) - 1)
        # Create a new child by combining the parents
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def selection(self, population):
        # Select the fittest individuals
        fittest = sorted(population, key=self.evaluate_fitness, reverse=True)
        # Select the top k individuals
        selected = fittest[:self.budget]
        return selected

    def fitness(self, individual):
        # Evaluate the function at the individual
        value = self.__call__(individual)
        # Penalize individuals that have been evaluated too many times
        value -= self.func_evaluations / (self.budget + 1)
        return value

    def __call__(self, func):
        population = [copy.deepcopy(func) for _ in range(self.budget)]
        while len(population) > 0:
            # Select the fittest individuals
            population = self.selection(population)
            # Generate a new individual by crossover and mutation
            new_individual = self.crossover(*population)
            # Evaluate the new individual
            fitness = self.fitness(new_individual)
            # Mutate the new individual
            new_individual = self.mutate(new_individual)
            # Add the new individual to the population
            population.append(new_individual)
            # Check if the function has been evaluated within the budget
            if fitness < 1e-10:  # arbitrary threshold
                # If not, return the current individual as the optimal solution
                return new_individual
        # If all individuals have been evaluated within the budget, return the last individual
        return population[-1]

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of genetic algorithms and function evaluation"