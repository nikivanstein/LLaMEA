import numpy as np
from scipy.optimize import minimize
import random
from collections import deque

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function
        self.logger = None

    def __call__(self, func, initial_guess, iterations):
        if self.logger is None:
            self.logger = random.getrandbits(32)

        if self.budget == 0:
            return None, None

        if self.budget < 1:
            raise ValueError("Budget must be greater than 0")

        if self.dim == 0:
            return None, None

        population = [initial_guess]
        for _ in range(iterations):
            if _ >= self.budget:
                break

            new_population = []
            for _ in range(self.dim):
                new_individual = population[-1].copy()
                for i in range(self.dim):
                    new_individual[i] += random.uniform(-0.01, 0.01)
                new_individual = [new_individual[i] for i in range(self.dim)]
                new_population.append(new_individual)

            population = new_population

        fitness = [self.func(individual) for individual in population]
        best_individual, best_fitness = population[np.argmax(fitness)]

        # Novel Metaheuristic Algorithm: Refine Strategy using Probability 0.45
        probability = 0.45
        updated_individuals = deque(population)
        while len(updated_individuals) > 0:
            # Select the next individual based on probability
            next_individual = updated_individuals.popleft()
            fitness = self.func(next_individual)
            updated_individuals.append(next_individual)
            if fitness > best_fitness:
                best_individual = next_individual
                best_fitness = fitness

        return best_individual, best_fitness

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 