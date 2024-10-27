import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func, budget=1000):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < budget:
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

    def novel_metaheuristic_algorithm(self, func, budget=1000, mutation_rate=0.1, cooling_rate=0.99):
        # Initialize the population with random individuals
        population = self.generate_population(budget)

        # Evaluate the fitness of each individual
        fitnesses = [individual[1] for individual in population]

        # Select the fittest individuals
        fittest_individuals = self.select_fittest(population, fitnesses)

        # Evolve the population
        for _ in range(100):
            # Select the fittest individuals
            fittest_individuals = self.select_fittest(fittest_individuals, fitnesses)

            # Generate a new population
            new_population = self.generate_population(budget)

            # Evaluate the fitness of each individual
            fitnesses = [individual[1] for individual in new_population]

            # Select the fittest individuals
            fittest_individuals = self.select_fittest(new_population, fitnesses)

            # Mutate the fittest individuals
            mutated_individuals = []
            for individual in fittest_individuals:
                mutated_individual = individual.copy()
                if random.random() < mutation_rate:
                    mutated_individual[0] += random.uniform(-1, 1)
                    mutated_individual[1] += random.uniform(-1, 1)
                mutated_individuals.append(mutated_individual)

            # Update the population
            population = new_population

        # Evaluate the fitness of each individual
        fitnesses = [individual[1] for individual in population]

        # Select the fittest individuals
        fittest_individuals = self.select_fittest(population, fitnesses)

        # Return the fittest individual
        return fittest_individuals[0]

    def generate_population(self, budget):
        # Initialize the population with random individuals
        population = []
        for _ in range(budget):
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = self.func(point)
            # Add the point and evaluation to the population
            population.append((point, evaluation))
        return population

    def select_fittest(self, population, fitnesses):
        # Select the fittest individuals
        fittest_individuals = []
        for _ in range(len(population)):
            # Select the individual with the highest fitness
            fittest_individual = max(population, key=lambda individual: individual[1])
            # Add the fittest individual to the list
            fittest_individuals.append(fittest_individual)
        return fittest_individuals

# One-line description: Novel metaheuristic algorithm for black box optimization using a combination of random walk and linear interpolation.