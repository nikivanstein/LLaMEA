import numpy as np
import random
from scipy.optimize import differential_evolution

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

    def novel_metaheuristic(self, func, budget, dim):
        # Define the mutation and crossover operators
        def mutation(individual):
            # Randomly select two points in the search space
            p1, p2 = random.sample(range(self.search_space[0], self.search_space[1]), 2)
            # Perform linear interpolation between the points
            new_individual = [p1 + (p2 - p1) * t / 10 for t in [0, 1]]
            # Check if the new individual is within the search space
            if np.all(new_individual >= self.search_space[0]) and np.all(new_individual <= self.search_space[1]):
                return new_individual
            else:
                return individual

        def crossover(parent1, parent2):
            # Select a random crossover point
            crossover_point = random.randint(1, self.dim)
            # Split the parents into two parts
            child1 = parent1[:crossover_point]
            child2 = parent2[crossover_point:]
            # Perform crossover between the two parts
            child1 = np.concatenate((child1, child2))
            # Check if the child is within the search space
            if np.all(child1 >= self.search_space[0]) and np.all(child1 <= self.search_space[1]):
                return child1
            else:
                return parent1, parent2

        # Initialize the population
        population = [func(np.random.uniform(self.search_space[0], self.search_space[1])) for _ in range(100)]

        # Evolve the population using differential evolution
        for _ in range(100):
            # Evaluate the fitness of each individual
            fitness = [func(individual) for individual in population]
            # Select the fittest individuals
            fittest = np.argsort(fitness)[-5:]
            # Perform mutation and crossover on the fittest individuals
            mutated_population = []
            for individual in fittest:
                mutated_individual = mutation(individual)
                mutated_population.append(mutated_individual)
            mutated_population = [func(individual) for individual in mutated_population]
            # Perform crossover on the mutated population
            for i in range(len(mutated_population) // 2):
                parent1, parent2 = mutated_population[i], mutated_population[len(mutated_population) - 1 - i]
                child1, child2 = crossover(parent1, parent2)
                mutated_population[i] = child1
                mutated_population[len(mutated_population) - 1 - i] = child2
            # Replace the old population with the new one
            population = mutated_population

        # Return the best individual
        best_individual = np.argmax(fitness)
        return best_individual, fitness[best_individual]

# One-line description: Novel metaheuristic algorithm for black box optimization using a combination of random walk and linear interpolation.

# Code: