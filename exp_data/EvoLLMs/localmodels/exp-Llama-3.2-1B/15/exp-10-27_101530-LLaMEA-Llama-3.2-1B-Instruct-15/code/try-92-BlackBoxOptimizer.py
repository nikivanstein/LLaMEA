import numpy as np
import random
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

    def novel_metaheuristic_algorithm(self, func, initial_population, mutation_rate, population_size, budget):
        # Initialize the population with random points in the search space
        population = initial_population

        # Evaluate the fitness of each individual in the population
        fitnesses = np.array([func(individual) for individual in population])

        # Initialize the best individual and its fitness
        best_individual = population[0]
        best_fitness = fitnesses[0]

        # Run the stochastic local search algorithm
        for _ in range(budget):
            # Select a random subset of individuals from the population
            subset = np.random.choice(len(population), size=population_size, replace=False)

            # Evaluate the fitness of each individual in the subset
            subset_fitnesses = np.array([func(individual) for individual in subset])

            # Calculate the average fitness of the subset
            avg_fitness = np.mean(subset_fitnesses)

            # Select the individual with the highest average fitness
            selected_individual = np.argmax(subset_fitnesses)

            # Mutate the selected individual
            mutated_individual = list(population[selected_individual])
            if random.random() < mutation_rate:
                mutated_individual[0] += random.uniform(-1, 1)
                mutated_individual[1] += random.uniform(-1, 1)

            # Replace the selected individual with the mutated individual
            population[selected_individual] = mutated_individual

            # Update the best individual and its fitness
            if np.mean(subset_fitnesses) > best_fitness:
                best_individual = selected_individual
                best_fitness = np.mean(subset_fitnesses)

        # Return the best individual found
        return best_individual