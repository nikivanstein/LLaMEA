import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = np.inf

    def initialize_population(self):
        # Generate a population of random solutions
        solutions = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        return solutions

    def __call__(self, func):
        # Optimize the black box function using the current population
        for _ in range(self.budget):
            # Evaluate the function at each solution in the current population
            fitness_scores = func(solutions)
            # Select the fittest solutions
            fittest_solutions = self.population[np.argsort(-self.fitness_scores)]
            # Refine the population using the fittest solutions
            self.population = fittest_solutions[:self.population_size // 2]
            # Update the fitness scores and the best solution
            self.fitness_scores = np.array([func(solution) for solution in self.population])
            self.best_solution = self.population[np.argmin(self.fitness_scores)]
            self.best_fitness = min(self.best_fitness, np.min(self.fitness_scores))
            # Refine the strategy by changing 10% of the solutions
            self.refine_strategy()

    def refine_strategy(self):
        # Refine the strategy by changing 10% of the solutions
        for i in range(self.population_size):
            if random.random() < 0.1:
                self.population[i] = random.uniform(-5.0, 5.0)

    def __str__(self):
        return f"BlackBoxOptimizer(budget={self.budget}, dim={self.dim})"

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 