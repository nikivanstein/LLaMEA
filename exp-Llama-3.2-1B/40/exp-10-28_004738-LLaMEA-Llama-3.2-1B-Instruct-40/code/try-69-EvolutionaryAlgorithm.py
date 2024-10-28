import random
import math
import numpy as np

class EvolutionaryAlgorithm:
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

    def __call__(self, func, population_size=100, mutation_rate=0.1, selection_rate=0.5, bounds=None):
        # Initialize population
        if bounds is None:
            bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        population = self.generate_population(population_size, bounds)

        # Evaluate fitness and select individuals
        fitness = self.evaluate_fitness(population)
        selected = np.random.choice(population_size, size=population_size, replace=True, p=fitness / np.sum(fitness))

        # Evolve population
        for _ in range(self.budget):
            # Select parents
            parents = self.select_parents(selected, fitness)

            # Create new offspring
            offspring = self.create_offspring(parents, mutation_rate)

            # Evaluate fitness and select new parents
            new_fitness = self.evaluate_fitness(offspring)
            new_selected = np.random.choice(population_size, size=population_size, replace=True, p=new_fitness / np.sum(new_fitness))

            # Replace old population with new
            population = new_selected

        return population

    def generate_population(self, population_size, bounds):
        return [random.uniform(bounds[0], bounds[1]) for _ in range(population_size)]

    def evaluate_fitness(self, population):
        fitness = np.zeros(population_size)
        for individual in population:
            func = self.funcs[individual]
            fitness[individual] = self.evaluate_func(func)
        return fitness

    def select_parents(self, selected, fitness):
        parents = []
        for i, individual in enumerate(selected):
            parent = individual
            fitness_value = fitness[i]
            cumulative_fitness = 0
            for j, selected_j in enumerate(selected):
                if selected_j == i:
                    continue
                cumulative_fitness += fitness_j
                if cumulative_fitness >= fitness_value:
                    break
            parents.append(parent)
        return np.array(parents)

    def create_offspring(self, parents, mutation_rate):
        offspring = []
        for _ in range(len(parents)):
            parent = parents[_]
            child = parent.copy()
            for _ in range(self.dim):
                if random.random() < mutation_rate:
                    child[_] = random.uniform(bounds[0], bounds[1])
            offspring.append(child)
        return offspring

    def evaluate_func(self, func):
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
    return EvolutionaryAlgorithm(budget, len(bounds)).__call__(func, population_size=100, mutation_rate=0.1, selection_rate=0.5, bounds=bounds)

# Description: Black Box Optimization using BBOB
# Code: 