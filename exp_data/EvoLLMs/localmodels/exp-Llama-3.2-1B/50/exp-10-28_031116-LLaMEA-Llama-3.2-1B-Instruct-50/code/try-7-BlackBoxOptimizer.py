import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.fitness_values = self.calculate_fitness_values()

    def initialize_population(self):
        # Initialize the population with random solutions
        population = []
        for _ in range(self.population_size):
            dim = random.randint(1, self.dim)
            solution = np.random.uniform(-5.0, 5.0, dim)
            population.append(solution)
        return population

    def calculate_fitness_values(self):
        # Calculate the fitness values for each solution
        fitness_values = {}
        for solution in self.population:
            func = self.budget(solution)
            fitness_values[solution] = func
        return fitness_values

    def __call__(self, func):
        # Optimize the black box function using the current population
        while len(self.population) > 0:
            # Select the fittest solution
            fittest_solution = self.population[np.argmax(self.fitness_values)]

            # Refine the fittest solution using the probability 0.45
            refinement = fittest_solution + (np.random.uniform(-0.1, 0.1),)

            # Check if the new solution is within the search space
            if np.abs(refinement[0] - 5.0) < 0.1 and np.abs(refinement[1] - 5.0) < 0.1:
                # Update the fittest solution
                self.population.remove(fittest_solution)
                self.population.append(refinement)
                self.fitness_values[refinement] = func
                return refinement

    def budget(self, solution):
        # Evaluate the black box function for the given solution
        func = self.budget(solution)
        return func

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using population refinement strategy
# 
# Parameters:
#   budget: maximum number of function evaluations
#   dim: dimensionality of the search space
# 
# Returns:
#   fittest solution: optimized solution with high fitness value