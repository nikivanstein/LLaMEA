import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population_size = 100
        self.population_size_min = 10
        self.population_size_max = 1000

    def __call__(self, func, population=None):
        if population is None:
            population = self.initialize_population()

        # Evaluate the function for each individual in the population
        fitnesses = [self.evaluate_fitness(individual, func, self.budget) for individual in population]

        # Select the fittest individuals
        fittest_individuals = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)[:self.population_size]

        # Select new individuals using the fittest individuals
        new_individuals = []
        while len(new_individuals) < self.population_size:
            parent1, parent2 = random.sample(fittest_individuals, 2)
            child = self.crossover(parent1, parent2)
            new_individuals.append(child)

        # Replace the old population with the new ones
        population = new_individuals

        # Check if the population is large enough
        if len(population) < self.population_size_min:
            population = self.initialize_population()

        # Evaluate the function for each individual in the population
        fitnesses = [self.evaluate_fitness(individual, func, self.budget) for individual in population]

        # Select the fittest individuals
        fittest_individuals = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)[:self.population_size]

        # Select new individuals using the fittest individuals
        new_individuals = []
        while len(new_individuals) < self.population_size:
            parent1, parent2 = random.sample(fittest_individuals, 2)
            child = self.mutation(parent1, parent2)
            new_individuals.append(child)

        # Replace the old population with the new ones
        population = new_individuals

        # Return the fittest individual
        return fittest_individuals[0][0]

    def initialize_population(self):
        return [np.random.uniform(self.search_space) for _ in range(self.population_size)]

    def evaluate_fitness(self, individual, func, budget):
        num_evaluations = min(budget, individual.size)
        func_evaluations = individual.size
        individual = individual[:]

        for _ in range(num_evaluations):
            point = np.random.choice(self.search_space)
            value = func(point)
            if value < 1e-10:  # arbitrary threshold
                return point
            individual = np.vstack((individual, point))

        return individual

    def crossover(self, parent1, parent2):
        index1, index2 = random.sample(range(parent1.size), 2)
        child = np.concatenate((parent1[:index1], parent2[index2:]))
        return child

    def mutation(self, parent1, parent2):
        index1, index2 = random.sample(range(parent1.size), 2)
        child = np.concatenate((parent1[:index1], parent2[index2:]))
        return child

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"