import numpy as np
import random
import math

class EvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            dim = np.random.uniform(-5.0, 5.0, self.dim)
            func = self.budget(dim)
            population.append((func, random.random()))
        return population

    def __call__(self, func):
        # Evaluate the function using the current population
        best_func = None
        best_score = -np.inf
        for func, _ in self.population:
            score = func(func)
            if score > best_score:
                best_func = func
                best_score = score
        # Select the best function and create a new individual
        best_func, _ = self.population[0]
        new_func = best_func(best_func)
        new_individual = np.random.uniform(-5.0, 5.0, self.dim)
        # Adapt the new individual using mutation and selection
        mutation_prob = 0.01
        new_individual = new_individual + random.uniform(-mutation_prob, mutation_prob)
        # Add the new individual to the population
        self.population.append((new_func, new_individual))
        return new_func

    def select(self):
        # Select the best individual using tournament selection
        winners = []
        for _ in range(self.population_size):
            winner = random.choice(self.population)
            winners.append(winner[1])
        # Refine the selected solution using adaptive mutation and selection
        mutation_prob = 0.01
        for _ in range(self.population_size):
            winner = random.choice(winners)
            if winner!= self.population[0][1]:
                winner = random.choice(self.population)
                mutation_prob = max(0.01, mutation_prob * 0.9)
                winner = (winner[0] + random.uniform(-mutation_prob, mutation_prob), winner[1])
        return winner