import random
import numpy as np
import copy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.population_size = 100
        self.mutation_rate = 0.01

    def __call__(self, func):
        # Initialize the population with random individuals
        population = [copy.deepcopy(func) for _ in range(self.population_size)]

        # Define the fitness function
        def fitness(individual):
            # Evaluate the function at the individual
            func_value = individual[0]
            for dim, value in zip(individual[1:], func_value):
                if value < 0:
                    value = -value
                elif value > 0:
                    value = -value + 1
            return np.mean(np.abs(value))

        # Evaluate the fitness of each individual
        fitness_values = [fitness(individual) for individual in population]

        # Select the fittest individuals
        fittest_indices = np.argsort(fitness_values)[-self.population_size//2:]
        fittest_individuals = [population[i] for i in fittest_indices]

        # Create a new generation
        new_population = []
        for _ in range(self.population_size):
            # Select two parents from the fittest individuals
            parent1, parent2 = random.sample(fittest_individuals, 2)
            # Crossover (recombination) the parents
            child = [(parent1[dim] + parent2[dim]) / 2 for dim in range(self.dim)]
            # Mutate the child
            for i in range(self.dim):
                if random.random() < self.mutation_rate:
                    child[i] += random.uniform(-1, 1)
            # Add the child to the new population
            new_population.append(child)

        # Replace the old population with the new one
        population = new_population

        # Update the best individual
        best_individual = max(population, key=fitness)
        best_individual = [best_individual[dim] for dim in range(self.dim)]

        # Evaluate the fitness of the best individual
        best_fitness = fitness(best_individual)

        # Update the budget and the best individual
        if self.func_evaluations < self.budget:
            return best_individual
        else:
            return self.search_space[0], self.search_space[1]

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization using Evolutionary Strategies