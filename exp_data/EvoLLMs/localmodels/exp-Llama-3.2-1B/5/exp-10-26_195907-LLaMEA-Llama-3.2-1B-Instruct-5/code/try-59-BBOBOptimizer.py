import random
import numpy as np
import math
from scipy.optimize import minimize

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        while True:
            # Initialize the population with random individuals
            population = self.generate_population(self.budget)

            # Evaluate the fitness of each individual in the population
            fitnesses = [self.evaluate_fitness(individual, func) for individual in population]

            # Select the fittest individuals to reproduce
            fittest_individuals = [individual for individual, fitness in zip(population, fitnesses) if fitness == max(fitnesses)]

            # Create new offspring by combining the fittest individuals
            offspring = []
            while len(offspring) < self.budget:
                parent1, parent2 = random.sample(fittest_individuals, 2)
                child = self.crossover(parent1, parent2)
                offspring.append(child)

            # Evaluate the fitness of the new offspring
            new_fitnesses = [self.evaluate_fitness(individual, func) for individual in offspring]

            # Select the fittest offspring to reproduce
            new_fittest_individuals = [individual for individual, fitness in zip(offspring, new_fitnesses) if fitness == max(new_fitnesses)]

            # Replace the old population with the new population
            population = new_fittest_individuals

            # Refine the search space by sampling the fittest individuals
            self.search_space = self.generate_population(self.budget)

            # If the search space is too small, sample from the population instead
            if len(self.search_space) < self.budget * 0.05:
                self.search_space = np.random.choice(self.search_space, size=(dim, 2), replace=True)

# Helper functions
def generate_population(budget):
    population = []
    for _ in range(budget):
        individual = np.random.uniform(-5.0, 5.0, size=self.dim)
        population.append(individual)
    return population

def evaluate_fitness(individual, func):
    return func(individual)

def crossover(parent1, parent2):
    return np.vstack((parent1[:len(parent1)//2], parent2[len(parent2)//2:]))

def minimizeBBOB(func, bounds, initial_guess, budget):
    return minimize(func, initial_guess, args=(bounds,), method="SLSQP", bounds=bounds, options={"maxiter": 1000})