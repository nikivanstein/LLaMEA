import random
import numpy as np
import math
from collections import deque

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0
        self.population_size = 100
        self.population = deque(maxlen=self.population_size)

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

    def mutate(self, individual):
        # Select a random mutation point
        mutation_point = random.randint(0, self.dim - 1)
        # Swap the element at the mutation point with a random element from the search space
        individual[mutation_point], individual[mutation_point] = random.choice(list(zip(individual[mutation_point:], individual[mutation_point:]))) + [individual[mutation_point]]
        return individual

    def crossover(self, parent1, parent2):
        # Select two parents and create a child by crossover
        child = parent1[:self.dim // 2] + parent2[self.dim // 2:]
        return child

    def __next__(self):
        # Select an individual from the population using roulette wheel selection
        probabilities = [1 / self.population_size for _ in self.population]
        cumulative_probabilities = [sum(probabilities[:i + 1]) for i in range(self.population_size)]
        r = random.random()
        for i, probability in enumerate(cumulative_probabilities):
            if r < probability:
                return self.population[i]
        raise StopIteration

    def __next_population(self):
        # Select the next population size using elitism
        new_population = list(self.population)
        for _ in range(self.population_size - len(new_population)):
            new_population.append(self.__next__())
        return new_population

    def optimize(self, func):
        # Initialize the population with random individuals
        self.population = [self.__next__() for _ in range(self.population_size)]
        # Run the optimization algorithm for a fixed number of iterations
        for _ in range(100):
            # Select the next population
            next_population = self.__next_population()
            # Evaluate the fitness of each individual
            fitness = [func(individual) for individual in next_population]
            # Select the fittest individuals
            self.population = self.__next__(self.population_size - len(next_population))
            # Replace the old population with the new one
            self.population = next_population
        # Evaluate the fitness of each individual
        fitness = [func(individual) for individual in self.population]
        # Select the fittest individual
        best_individual = self.__next__(self.population_size)
        # Return the best individual and its fitness
        return best_individual, fitness[-1]