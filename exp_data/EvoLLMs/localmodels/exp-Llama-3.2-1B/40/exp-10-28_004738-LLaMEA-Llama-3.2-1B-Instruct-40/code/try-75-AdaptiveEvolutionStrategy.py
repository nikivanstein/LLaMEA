import random
import numpy as np

class AdaptiveEvolutionStrategy:
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

    def __call__(self, func, x0, bounds, population_size, mutation_rate, exploration_rate):
        # Initialize population with random individuals
        population = [x0 for _ in range(population_size)]

        # Evaluate fitness of each individual
        fitnesses = [func(individual, bounds) for individual in population]

        # Select parents using tournament selection
        parents = []
        for _ in range(population_size // 2):
            parent1, parent2 = random.sample(population, 2)
            if random.random() < exploration_rate:
                parent1, parent2 = parent2, parent1
            fitnesses1, fitnesses2 = zip(*sorted(zip(fitnesses1, fitnesses2), key=lambda x: x[0]))
            if random.random() < 0.5:
                parents.append(parent1)
            else:
                parents.append(parent2)

        # Create new generation
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.sample(parents, 2)
            if random.random() < exploration_rate:
                parent1, parent2 = parent2, parent1
            child = (parent1 + parent2) / 2
            for _ in range(self.budget):
                child = func(child, bounds)
                if child < bounds[0]:
                    child = bounds[0]
                elif child > bounds[1]:
                    child = bounds[1]
                if random.random() < 0.5:
                    child = random.uniform(bounds[0], bounds[1])
                if random.random() < 0.2:
                    child = random.uniform(bounds[0], bounds[1])
                if random.random() < 0.4:
                    child = random.uniform(bounds[0], bounds[1])
            new_population.append(child)

        # Evaluate fitness of new generation
        fitnesses = [func(individual, bounds) for individual in new_population]

        # Select best individuals for replacement
        best_individuals = sorted(zip(fitnesses, population), reverse=True)[:population_size // 2]
        new_population = [individual for fitness, individual in best_individuals]

        # Replace old population with new population
        population = new_population

        return population

def f(x):
    return x**2 + 0.5*x + 0.1

def f_prime(x):
    return 2*x + 0.5

def f_double_prime(x):
    return 2

def f_double_prime_prime(x):
    return 4

def bbo_opt(func, x0, bounds, budget):
    return AdaptiveEvolutionStrategy(budget, len(bounds)).__call__(func, x0, bounds, 100, 0.1, 0.4)

# Description: Adaptive Evolution Strategy for Black Box Optimization
# Code: 