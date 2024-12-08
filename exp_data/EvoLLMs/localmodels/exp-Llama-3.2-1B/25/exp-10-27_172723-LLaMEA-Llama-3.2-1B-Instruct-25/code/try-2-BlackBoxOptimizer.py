import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

    def evolve(self, func, budget):
        # Initialize the population with random points in the search space
        population = self.generate_population(budget)

        # Evolve the population using differential evolution
        for _ in range(100):  # Run for 100 generations
            # Evaluate the fitness of each individual in the population
            fitness = [self.__call__(individual, func) for individual in population]

            # Select the fittest individuals
            fittest = np.argsort(fitness)[:self.budget]

            # Mutate the fittest individuals with a probability of 0.25
            mutated = [individual.copy() for individual in fittest]
            for _ in range(self.budget):
                if random.random() < 0.25:
                    mutated[_] = random.uniform(self.search_space)

            # Replace the least fit individuals with the mutated ones
            population[fittest] = mutated

        # Return the fittest individual
        return population[np.argmax(fitness)]

    def generate_population(self, budget):
        # Generate a population of random points in the search space
        population = np.random.choice(self.search_space, size=(budget, self.dim), replace=False)
        return population

# One-line description: "Evolutionary Algorithm: A novel metaheuristic algorithm that efficiently solves black box optimization problems using evolutionary strategies"