import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def novel_metaheuristic(self, func, population_size, mutation_rate, mutation_cost, bounds, initial_population):
        # Initialize the population
        population = initial_population

        # Run the genetic algorithm for a fixed number of generations
        for _ in range(100):
            # Evaluate the fitness of each individual
            fitness = [func(individual) for individual in population]

            # Select the fittest individuals
            fittest = np.argsort(fitness)[-population_size:]

            # Create a new population by crossover and mutation
            new_population = []
            for _ in range(population_size):
                parent1, parent2 = random.sample(fittest, 2)
                child = (parent1[0] + 2 * parent2[0]) / 2, (parent1[1] + 2 * parent2[1]) / 2
                if random.random() < mutation_rate:
                    child[0] += np.random.uniform(-mutation_cost, mutation_cost)
                    child[1] += np.random.uniform(-mutation_cost, mutation_cost)
                new_population.append(child)

            # Replace the old population with the new one
            population = new_population

        # Return the best individual in the final population
        return population[0]

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# Refines the strategy by introducing crossover and mutation to adapt to changing fitness landscapes
# Parameters: budget, dim, mutation_rate, mutation_cost, bounds, initial_population
# Returns: best individual in the final population