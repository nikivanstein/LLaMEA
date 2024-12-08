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

    def novel_metaheuristic(self, budget, dim, iterations=100):
        # Initialize the population with random points in the search space
        population = np.random.uniform(self.search_space[0], self.search_space[1], (dim, budget))
        
        # Define the mutation function
        def mutate(individual):
            # Select a random dimension
            dim_index = random.randint(0, dim - 1)
            # Randomly select a value within the search space
            new_value = random.uniform(self.search_space[0], self.search_space[1])
            # Replace the value in the individual with the new value
            individual[dim_index] = new_value
            return individual
        
        # Define the selection function
        def select(population):
            # Calculate the fitness of each individual
            fitness = np.array([func(individual) for individual in population])
            # Select the fittest individuals
            selected = np.argsort(fitness)[-int(0.15 * budget):]
            return selected
        
        # Define the crossover function
        def crossover(parent1, parent2):
            # Select a random crossover point
            crossover_point = random.randint(0, dim - 1)
            # Create a child by combining the two parents
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            return child
        
        # Define the mutation operator
        def mutate_individual(individual):
            # Apply mutation to each individual
            for _ in range(10):
                mutate(individual)
            return individual
        
        # Run the optimization algorithm
        for _ in range(iterations):
            # Select the fittest individuals
            selected = select(population)
            # Create a new population by crossover and mutation
            new_population = np.array([crossover(parent, mutate_individual(individual)) for parent, individual in zip(selected, population)])
            # Replace the old population with the new one
            population = new_population
        
        # Return the best individual in the new population
        return new_population[0]