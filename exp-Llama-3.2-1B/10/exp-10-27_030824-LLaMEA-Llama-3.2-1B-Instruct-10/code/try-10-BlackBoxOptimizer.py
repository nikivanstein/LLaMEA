import random
import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def optimize(self, func, budget, iterations=1000):
        # Initialize the population with random points
        population = [self.__call__(func) for _ in range(100)]

        # Define the mutation function
        def mutate(individual, budget):
            # Randomly change a single element in the individual
            return individual[:], individual + [random.uniform(self.search_space[0], self.search_space[1])]

        # Define the selection function
        def select(population, budget):
            # Select the top k individuals based on their fitness
            fitnesses = [individual[1] for individual in population]
            return [individual for _, individual in sorted(zip(fitnesses, population), reverse=True)[:budget]]

        # Define the crossover function
        def crossover(parent1, parent2):
            # Select a random crossover point
            crossover_point = random.randint(1, len(parent1) - 1)
            # Create a new child by combining the two parents
            child = parent1[:crossover_point] + parent2[crossover_point:]
            return child

        # Run the evolutionary algorithm
        for _ in range(iterations):
            # Select the fittest individuals
            population = select(population, budget)

            # Mutate the selected individuals
            population = [mutate(individual, budget) for individual in population]

            # Crossover the selected individuals
            population = [crossover(parent1, parent2) for parent1, parent2 in zip(population, population[1:])]

        # Return the fittest individual
        return population[0]

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.