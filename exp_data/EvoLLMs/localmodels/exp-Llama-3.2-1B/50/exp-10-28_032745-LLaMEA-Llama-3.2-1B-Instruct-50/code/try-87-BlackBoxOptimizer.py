import numpy as np
from scipy.optimize import minimize
import random
import operator

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function

    def __call__(self, func, initial_guess, iterations):
        # Define a fitness function that returns the negative value to minimize
        def fitness(individual):
            return -self.func(individual)

        # Initialize the population with random individuals
        population = [initial_guess for _ in range(self.budget)]

        # Define the selection function
        def selection(population, budget, dim):
            # Select the fittest individuals
            fitnesses = [fitness(individual) for individual in population]
            selected_indices = np.argsort(fitnesses)[:self.budget]
            return [population[i] for i in selected_indices]

        # Define the crossover function
        def crossover(parent1, parent2):
            # Select two parents based on the probability 0.45
            parent1, parent2 = selection(population, self.budget, self.dim)
            # Perform crossover
            child = [random.uniform(self.search_space[i][0], self.search_space[i][1]) for i in range(self.dim)]
            for i in range(self.dim):
                if random.random() < 0.45:
                    child[i] = parent1[i]
                else:
                    child[i] = parent2[i]
            return child

        # Define the mutation function
        def mutation(individual):
            # Select a random index and mutate the individual
            index = random.randint(0, self.dim - 1)
            individual[index] = random.uniform(self.search_space[index][0], self.search_space[index][1])
            return individual

        # Evolve the population for the specified number of iterations
        for _ in range(iterations):
            # Perform crossover and mutation
            population = [crossover(parent, mutation(individual)) for parent, individual in zip(population, population)]
            # Replace the least fit individuals with the new population
            population = selection(population, self.budget, self.dim)

        # Return the fittest individual
        return population[0], fitness(population[0])

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using evolutionary strategies