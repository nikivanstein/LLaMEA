import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        # Initialize the population with random solutions
        population = []
        for _ in range(self.budget):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population

    def __call__(self, func):
        # Optimize the function using the current population
        def evaluate(func, solution):
            return func(solution)

        # Select the fittest solutions
        fittest_solutions = sorted(self.population, key=evaluate, reverse=True)[:self.budget]

        # Refine the population using the selected solutions
        new_population = []
        while len(new_population) < self.budget:
            # Select two parents from the fittest solutions
            parent1, parent2 = random.sample(fittest_solutions, 2)
            # Crossover (reproduce) the parents to create a new child
            child = self.crossover(parent1, parent2)
            # Mutate the child to introduce randomness
            child = self.mutate(child)
            # Add the child to the new population
            new_population.append(child)

        # Replace the old population with the new one
        self.population = new_population

    def crossover(self, parent1, parent2):
        # Perform crossover (reproduce) the parents
        child = parent1
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent2[i]
        return child

    def mutate(self, solution):
        # Mutate the solution to introduce randomness
        for i in range(self.dim):
            if random.random() < 0.1:
                solution[i] += random.uniform(-1.0, 1.0)
        return solution

    def select_parents(self):
        # Select the fittest solutions to reproduce
        fittest_solutions = sorted(self.population, key=evaluate, reverse=True)[:self.budget]
        return fittest_solutions

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 