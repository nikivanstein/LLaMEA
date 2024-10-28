import random
import numpy as np

class EvolutionStrategy:
    def __init__(self, budget, dim, mutation_prob=0.4):
        self.budget = budget
        self.dim = dim
        self.mutation_prob = mutation_prob
        self.funcs = self.generate_functions()

    def generate_functions(self):
        functions = []
        for _ in range(24):
            func = lambda x: random.uniform(-5.0, 5.0)
            functions.append(func)
        return functions

    def __call__(self, func, x0, bounds, population_size, mutation_rate):
        # Evaluate the fitness of the initial population
        fitnesses = [func(x, bounds, population_size) for x in x0]

        # Select the fittest individuals
        fittest = sorted(zip(fitnesses, x0), reverse=True)[:self.budget]

        # Create a new population by mutation and crossover
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.sample(fittest, 2)
            child = (parent1[0] + parent2[0]) / 2 + random.uniform(-0.1, 0.1)
            if random.random() < mutation_rate:
                child[0] = (child[0] + random.uniform(-1, 1)) / 2 + random.uniform(-0.1, 0.1)
            new_population.append(child)

        # Evaluate the new population
        new_fitnesses = [func(x, bounds, population_size) for x in new_population]

        # Replace the old population with the new one
        x0 = new_population
        fitnesses = new_fitnesses

        return x0, fitnesses

def f(x, bounds, population_size):
    return np.sum(x**2)

def f_prime(x, bounds, population_size):
    return np.sum(2*x)

def f_double_prime(x, bounds, population_size):
    return np.sum(2)

def f_double_prime_prime(x, bounds, population_size):
    return np.sum(4)

def bbo_opt(func, x0, bounds, budget, population_size, mutation_rate):
    strategy = EvolutionStrategy(budget, x0.shape[1], mutation_rate)
    best_individual, best_fitness = strategy.__call__(func, x0, bounds, population_size, mutation_rate)
    return best_individual, best_fitness

# Example usage:
x0 = np.random.uniform(-5.0, 5.0, (5,))
bounds = [(-5.0, 5.0) for _ in range(5)]
budget = 100
population_size = 100
mutation_rate = 0.05

best_individual, best_fitness = bbo_opt(f, x0, bounds, budget, population_size, mutation_rate)

# Print the result
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)