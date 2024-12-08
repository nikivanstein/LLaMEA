import numpy as np
from scipy.optimize import minimize
import random

class BBOOpt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = None
        self.fitnesses = None
        self.func = None

    def __call__(self, func):
        self.func = func
        self.population = self.generate_population(self.budget)
        self.fitnesses = self.calculate_fitnesses()
        return self.optimize()

    def generate_population(self, budget):
        # Generate a population of candidate solutions
        # Each solution is a vector of dimensionality
        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(budget)]
        return population

    def calculate_fitnesses(self):
        # Calculate the fitness of each candidate solution
        # The fitness is a scalar value representing the quality of the solution
        fitnesses = [self.func(x) for x in self.population]
        return fitnesses

    def optimize(self):
        # Perform the optimization algorithm
        # The algorithm uses a greedy approach with a probability of 0.45
        # and a probability of 0.55 to refine the strategy
        if random.random() < 0.45:
            # Use the current population to find the best candidate solution
            best_solution = self.population[np.argmax(self.fitnesses)]
        else:
            # Refine the strategy by sampling from the current population
            idx = random.randint(0, len(self.population) - 1)
            best_solution = self.population[idx]

        # Refine the strategy by sampling from the current population
        idx = random.randint(0, len(self.population) - 1)
        best_solution = self.population[idx]

        # Update the population with the best candidate solution
        # and its corresponding fitness
        new_population = self.population[:idx] + [best_solution] + self.population[idx+1:]
        self.population = new_population
        self.fitnesses = self.calculate_fitnesses()

        return best_solution, self.fitnesses

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 