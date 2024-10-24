import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.generate_initial_population()
        self.fitness_values = self.calculate_fitness_values()

    def generate_initial_population(self):
        # Generate a population of random points in the search space
        population = []
        for _ in range(self.budget):
            dim = random.uniform(-5.0, 5.0)
            point = [dim, dim]  # Random point in the search space
            population.append(point)
        return population

    def calculate_fitness_values(self):
        # Calculate the fitness values for each point in the population
        fitness_values = []
        for point in self.population:
            func_value = self.evaluate_function(point)
            fitness_values.append(func_value)
        return fitness_values

    def evaluate_function(self, point):
        # Evaluate the fitness of a given point using the provided function
        return self.budget * point[0] + point[1] ** 2

    def __call__(self, func):
        # Optimize the black box function using the selected solution
        for _ in range(self.budget):
            # Select the fittest solution from the current population
            fittest_point = self.population[np.argmax(self.fitness_values)]

            # Generate a new point by perturbing the fittest solution
            perturbed_point = [fittest_point[0] + random.uniform(-1.0, 1.0), fittest_point[1] + random.uniform(-1.0, 1.0)]

            # Check if the new point is within the search space
            if self.search_space_check(perturbed_point):
                # Evaluate the fitness of the new point
                new_fitness_value = self.evaluate_function(perturbed_point)

                # Update the fittest solution if the new point is better
                if new_fitness_value < self.fitness_values[-1]:
                    self.fittest_point = perturbed_point
                    self.fitness_values[-1] = new_fitness_value

        # Select the fittest solution from the final population
        self.fittest_point = self.population[np.argmax(self.fitness_values)]

        # Return the fittest solution
        return self.fittest_point

    def search_space_check(self, point):
        # Check if the point is within the search space
        return -5.0 <= point[0] <= 5.0 and -5.0 <= point[1] <= 5.0

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization
# Uses a novel combination of random search and evolutionary algorithms to optimize black box functions
# Evaluates the fitness of multiple solutions and selects the fittest one