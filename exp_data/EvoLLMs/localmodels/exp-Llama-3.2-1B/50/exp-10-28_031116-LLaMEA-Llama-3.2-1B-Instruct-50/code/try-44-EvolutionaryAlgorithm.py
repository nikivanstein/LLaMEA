import numpy as np
import random

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitness_scores = []

    def __call__(self, func):
        # Initialize population with random solutions
        for _ in range(self.budget):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            self.population.append(solution)

        # Evaluate population and select fittest solutions
        fitness_scores = []
        for solution in self.population:
            func(solution)
            fitness_scores.append(np.mean(np.abs(func(solution))))

        # Select fittest solutions
        self.fittest_solutions = random.choices(self.population, weights=fitness_scores, k=self.budget)

        # Update population with fittest solutions
        for solution in self.fittest_solutions:
            self.population.remove(solution)
            self.population.append(solution)

    def mutate(self, solution):
        # Randomly mutate a solution
        mutated_solution = solution + np.random.uniform(-1.0, 1.0, self.dim)
        return mutated_solution

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents
        child = np.copy(parent1)
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent2[i]
        return child

    def __str__(self):
        return f"Population: {self.population}, Fitness Scores: {self.fitness_scores}"

# Description: Evolutionary Algorithm for Black Box Optimization
# Code: 