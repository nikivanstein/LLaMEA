import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, iterations=100):
        # Define the mutation probability
        mutation_prob = 0.1
        # Define the population size
        population_size = 100

        # Initialize the population with random points in the search space
        population = [np.random.uniform(self.search_space[0], self.search_space[1]) for _ in range(population_size)]

        # Evaluate the fitness of each individual in the population
        for _ in range(iterations):
            # Select the best individual based on its fitness
            best_individual = population[np.argmax([func(individual) for individual in population])]

            # Generate a new individual by refining the best individual
            new_individual = best_individual
            for _ in range(random.randint(1, self.dim)):
                # Randomly select two points in the search space
                point1 = np.random.uniform(self.search_space[0], self.search_space[1])
                point2 = np.random.uniform(self.search_space[0], self.search_space[1])

                # Calculate the linear interpolation of the two points
                interpolated_point = point1 + (point2 - point1) * mutation_prob

                # Refine the new individual by replacing one of the points with the interpolated point
                if random.random() < mutation_prob:
                    new_individual = interpolated_point

            # Evaluate the fitness of the new individual
            new_fitness = func(new_individual)

            # Update the best individual and its fitness
            if new_fitness > func(new_individual):
                best_individual = new_individual
                best_fitness = new_fitness

            # Update the population with the new individual
            population.append(new_individual)

        # Return the best individual and its fitness
        return best_individual, best_fitness

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.
# Code: 
# ```python
# import random
# import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, iterations=100):
        # Define the mutation probability
        mutation_prob = 0.1
        # Define the population size
        population_size = 100

        # Initialize the population with random points in the search space
        population = [np.random.uniform(self.search_space[0], self.search_space[1]) for _ in range(population_size)]

        # Evaluate the fitness of each individual in the population
        for _ in range(iterations):
            # Select the best individual based on its fitness
            best_individual = population[np.argmax([func(individual) for individual in population])]

            # Generate a new individual by refining the best individual
            new_individual = best_individual
            for _ in range(random.randint(1, self.dim)):
                # Randomly select two points in the search space
                point1 = np.random.uniform(self.search_space[0], self.search_space[1])
                point2 = np.random.uniform(self.search_space[0], self.search_space[1])

                # Calculate the linear interpolation of the two points
                interpolated_point = point1 + (point2 - point1) * mutation_prob

                # Refine the new individual by replacing one of the points with the interpolated point
                if random.random() < mutation_prob:
                    new_individual = interpolated_point

            # Evaluate the fitness of the new individual
            new_fitness = func(new_individual)

            # Update the best individual and its fitness
            if new_fitness > func(new_individual):
                best_individual = new_individual
                best_fitness = new_fitness

            # Update the population with the new individual
            population.append(new_individual)

        # Return the best individual and its fitness
        return best_individual, best_fitness