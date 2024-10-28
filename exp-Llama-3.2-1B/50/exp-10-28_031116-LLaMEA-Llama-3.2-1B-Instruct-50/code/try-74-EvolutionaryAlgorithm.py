import numpy as np
from scipy.optimize import minimize

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []

    def __call__(self, func):
        # Initialize the population with random solutions
        for _ in range(self.budget):
            self.population.append(self.generate_solution(func))

        # Evaluate the budget number of solutions
        solutions = [self.evaluate(func, self.population[i]) for i in range(self.budget)]

        # Select the fittest solutions
        self.population = self.select_fittest(solutions, self.budget)

        # Generate new solutions based on the fittest ones
        new_solutions = []
        for _ in range(self.budget):
            parent1, parent2 = np.random.choice(self.population, size=2, replace=False)
            child = self.crossover(parent1, parent2)
            new_solutions.append(child)

        # Evaluate the new solutions
        new_solutions = [self.evaluate(func, solution) for solution in new_solutions]

        # Select the fittest new solutions
        new_solutions = self.select_fittest(new_solutions, self.budget)

        # Replace the old population with the new ones
        self.population = new_solutions

        return self.evaluate(func, self.population[0])

    def generate_solution(self, func):
        # Generate a random solution within the search space
        return np.random.uniform(-5.0, 5.0, self.dim)

    def evaluate(self, func, solution):
        # Evaluate the function at the solution
        return func(solution)

    def select_fittest(self, solutions, budget):
        # Select the fittest solutions based on their fitness
        fittest = sorted(solutions, key=self.evaluate, reverse=True)[:budget]
        return fittest

    def crossover(self, parent1, parent2):
        # Perform crossover between the two parents
        child = parent1[:len(parent1)//2] + parent2[len(parent1)//2:]
        return child

    def mutate(self, solution):
        # Randomly mutate the solution
        return solution + np.random.uniform(-1.0, 1.0, self.dim)

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 