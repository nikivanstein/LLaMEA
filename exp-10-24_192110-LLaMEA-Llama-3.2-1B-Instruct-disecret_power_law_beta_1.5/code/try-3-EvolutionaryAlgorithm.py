import random
import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.initialize_population()
        self.fitness_scores = [0] * len(self.population)

    def initialize_population(self):
        # Initialize the population with random solutions
        population = []
        for _ in range(100):  # Initial population size
            solution = [random.uniform(-5.0, 5.0) for _ in range(self.dim)]
            population.append(solution)
        return population

    def __call__(self, func):
        # Evaluate the function for each solution in the population
        for solution in self.population:
            func(solution)
            # Refine the solution with probability 0.05
            if random.random() < 0.05:
                solution = self.refine_solution(solution)
        # Select the fittest solution
        fittest_solution = max(self.population, key=self.fitness_scores)
        return fittest_solution

    def refine_solution(self, solution):
        # Use the budget to select a new solution
        new_solution = solution.copy()
        for _ in range(self.budget):
            # Evaluate the function for the new solution
            func(new_solution)
            # Refine the new solution with probability 0.05
            if random.random() < 0.05:
                new_solution = self.refine_new_solution(new_solution)
        return new_solution

    def refine_new_solution(self, solution):
        # Use the budget to select a new solution
        new_solution = solution.copy()
        for _ in range(self.budget):
            # Evaluate the function for the new solution
            func(new_solution)
            # Refine the new solution with probability 0.05
            if random.random() < 0.05:
                new_solution = self.refine_new_solution(new_solution)
        return new_solution

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 