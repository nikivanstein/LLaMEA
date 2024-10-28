import random
import numpy as np

class BBOB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.funcs = self.generate_functions()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func):
        return func(random.uniform(-5.0, 5.0))

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget):
    # Define the mutation probability and the mutation direction
    mutation_prob = 0.4
    mutation_dir = np.array([1, 0])

    # Initialize the population
    population = [x0] * 100

    # Evolve the population
    for _ in range(budget):
        # Evaluate the fitness of each individual
        fitnesses = [self.f(individual, func, bounds) for individual, func, bounds in zip(population, func, bounds)]

        # Select the fittest individuals
        fittest_individuals = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:self.budget]

        # Create new individuals by mutation
        new_population = []
        for _ in range(self.budget):
            # Randomly select an individual
            individual, _ = fittest_individuals.pop(0)

            # Create a new individual by mutation
            new_individual = individual.copy()
            for _ in range(self.dim):
                # Randomly decide whether to mutate the new individual
                if random.random() < mutation_prob:
                    # Choose a random direction
                    dir = mutation_dir[np.random.randint(0, self.dim)]

                    # Mutate the new individual
                    new_individual[0] += dir * random.uniform(-1, 1)
                    new_individual[1] += dir * random.uniform(-1, 1)

            # Add the new individual to the new population
            new_population.append(new_individual)

        # Replace the old population with the new population
        population = new_population

    # Return the fittest individual in the new population
    return self.f(population[0], func, bounds)

# One-line description with the main idea
# Evolutionary Algorithm for Black Box Optimization using BBOB
# Code: 