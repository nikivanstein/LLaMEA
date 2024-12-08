import random
import numpy as np

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

    def novel_metaheuristic(self, population, budget):
        # Initialize the population with random points in the search space
        population = [np.random.uniform(self.search_space[0], self.search_space[1]) for _ in range(100)]

        # Evaluate the fitness of each individual in the population
        fitnesses = [self.__call__(func) for func in population]

        # Select the fittest individuals to reproduce
        fittest_individuals = sorted(zip(fitnesses, population), reverse=True)[:self.budget]

        # Create a new population by combining the fittest individuals
        new_population = []
        for _ in range(100):
            # Select two parents from the fittest individuals
            parent1, parent2 = random.sample(fittest_individuals, 2)
            # Combine the parents to create a new individual
            child = (parent1[0] + 2 * parent2[0]) / 3, (parent1[1] + 2 * parent2[1]) / 3
            # Add the child to the new population
            new_population.append(child)

        # Evaluate the fitness of the new population
        new_fitnesses = [self.__call__(func) for func in new_population]

        # Replace the old population with the new population
        population = new_population

        # Return the new population and its fitness
        return population, new_fitnesses

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.
