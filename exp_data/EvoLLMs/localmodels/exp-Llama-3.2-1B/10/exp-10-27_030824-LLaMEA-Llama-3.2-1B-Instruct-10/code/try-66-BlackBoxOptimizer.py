import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.population = None
        self.population_history = None
        self.search_space_history = None

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

    def evolve_population(self, population):
        # Select the fittest individuals
        fittest_individuals = sorted(population, key=lambda individual: individual.fitness, reverse=True)[:self.population_size]

        # Create a new population by applying mutation and crossover
        new_population = []
        for _ in range(self.population_size):
            parent1 = random.choice(fittest_individuals)
            parent2 = random.choice(fittest_individuals)
            child = self.mutate(parent1, parent2)
            new_population.append(child)

        # Replace the old population with the new one
        population[:] = new_population

        # Update the search space and fitness history
        self.search_space_history.append(self.search_space)
        self.population_history.append(population)

    def mutate(self, individual, parent1, parent2):
        # Randomly choose a mutation point
        mutation_point = random.randint(0, self.dim)

        # Apply random walk and linear interpolation
        mutated_individual = individual.copy()
        mutated_individual[mutation_point] = random.uniform(self.search_space[0], self.search_space[1])
        mutated_individual = self.linear_interpolate(mutated_individual, parent1, parent2, mutation_point)

        return mutated_individual

    def linear_interpolate(self, individual1, individual2, mutation_point, new_point):
        # Calculate the linear interpolation
        interpolated_individual = individual1.copy()
        interpolated_individual[mutation_point] = (individual1[mutation_point] + individual2[mutation_point]) / 2
        return interpolated_individual

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

# Code: