import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.population = []

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

    def select_individual(self):
        # Select the fittest individual from the population
        self.population = sorted(self.population, key=self.evaluate_fitness, reverse=True)[:self.budget]

    def mutate(self, individual):
        # Randomly swap two points in the individual
        if random.random() < 0.5:
            point1, point2 = random.sample(range(self.dim), 2)
            self.population[self.population.index(individual) - 1], self.population[self.population.index(individual)] = individual[point1], individual[point2]

    def crossover(self, parent1, parent2):
        # Select a random crossover point
        crossover_point = random.randint(1, self.dim - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def __next_generation(self, parent):
        # Create a new individual by crossover and mutation
        child = self.crossover(parent, parent)
        self.mutate(child)
        return child

    def fitness(self, individual):
        # Evaluate the fitness of the individual
        return np.sum(individual**2)

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.
# Code: 