import random
import numpy as np
import math

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

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

def linear_interpolation(point, lower_bound, upper_bound):
    return lower_bound + (upper_bound - lower_bound) * (point - lower_bound) / (upper_bound - lower_bound)

def random_walk(point, dimension):
    return point + random.uniform(-dimension, dimension)

def bounded_random_walk(point, lower_bound, upper_bound):
    return random.choice([lower_bound, upper_bound]) + random.uniform(0, upper_bound - lower_bound)

def calculate_fitness(point, func):
    return func(point)

def mutate(individual):
    if random.random() < 0.1:
        return random_walk(individual, self.dim)
    else:
        return bounded_random_walk(individual, self.search_space[0], self.search_space[1])

def mutate_and_evaluate(individual, func):
    new_individual = mutate(individual)
    return calculate_fitness(new_individual, func)

def generate_population(population_size, dim):
    return [random_walk(individual, dim) for individual in population_size]

def update_population(population, budget, func):
    new_population = []
    for _ in range(population_size):
        individual = generate_population(population_size, dim)
        fitness = calculate_fitness(individual, func)
        if fitness > 0.5:
            new_population.append(individual)
    return new_population

def evolve_population(population, budget, func):
    new_population = update_population(population, budget, func)
    return new_population

def evaluate_bboB(func, population, budget):
    new_population = evolve_population(population, budget, func)
    return evaluate_fitness(new_population, func)

def evaluate_fitness(population, func):
    fitness_values = []
    for individual in population:
        fitness_value = calculate_fitness(individual, func)
        fitness_values.append(fitness_value)
    return fitness_values

# Example usage:
def func1(x):
    return x**2

def func2(x):
    return np.sin(x)

budget = 1000
dim = 10
population = [np.random.uniform(-10, 10) for _ in range(100)]

fitness_values = evaluate_bboB(func1, population, budget)
fitness_values = evaluate_bboB(func2, population, budget)

print("Fitness values:", fitness_values)