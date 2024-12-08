import numpy as np
import random
from collections import deque

class MADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = None
        self.f = None
        self.g = None
        self.m = None
        self.m_history = []
        self.x_history = deque()

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Insufficient budget")

        # Initialize the current solution
        self.x = np.random.uniform(-5.0, 5.0, self.dim)
        self.f = func(self.x)

        # Initialize the mutation rate
        self.m = 0.1

        while self.budget > 0:
            # Evaluate the function at the current solution
            self.f = func(self.x)

            # Generate a new solution using differential evolution
            self.g = self.x + np.random.normal(0, 1, self.dim) * np.sqrt(self.f / self.budget)
            self.g = np.clip(self.g, -5.0, 5.0)

            # Evaluate the new solution
            self.g = func(self.g)

            # Check if the new solution is better
            if self.f < self.g:
                # Update the current solution
                self.x = self.g
                self.f = self.g

                # Update the mutation rate
                self.m = 0.1

            # Update the history
            self.x_history.append(self.x)
            self.m_history.append(self.m)

            # Decrease the budget
            self.budget -= 1

            # Check if the budget is zero
            if self.budget == 0:
                break

        return self.x

# Example usage:
def test_func(x):
    return np.sum(x**2)

made = MADE(1000, 10)
opt_x = made(__call__, test_func)
print(opt_x)

# Differential Evolution with Genetic Algorithm
def differential_evolution(func, bounds, mutation_rate, population_size):
    # Initialize the population
    population = [np.random.uniform(bounds[0], bounds[1], size=(population_size, len(bounds))) for _ in range(population_size)]

    # Initialize the mutation rate
    mutation_rate = mutation_rate / population_size

    # Evaluate the function at the initial population
    fitness = [func(individual) for individual in population]

    # Initialize the best solution and its fitness
    best_individual = population[fitness.index(max(fitness))]
    best_fitness = max(fitness)

    # Initialize the mutation counter
    mutation_counter = 0

    while mutation_counter < population_size:
        # Select the next individual
        individual = population[np.random.choice(population_size)]

        # Evaluate the function at the selected individual
        fitness = [func(individual[i]) for i in range(len(bounds))]

        # Generate a new solution using differential evolution
        new_individual = individual + np.random.normal(0, 1, len(bounds)) * np.sqrt(max(fitness) / population_size)
        new_individual = np.clip(new_individual, bounds[0], bounds[1])

        # Evaluate the new solution
        new_fitness = [func(new_individual[i]) for i in range(len(bounds))]

        # Check if the new solution is better
        if new_fitness.index(max(new_fitness)) > fitness.index(max(fitness)):
            # Update the best solution and its fitness
            best_individual = new_individual
            best_fitness = max(new_fitness)

            # Update the mutation counter
            mutation_counter += 1

        # Update the mutation rate
        mutation_rate = mutation_rate / mutation_counter

        # Update the population
        population = [individual if i not in [best_individual[i] for i in range(len(bounds))] else best_individual[i] for i in range(len(bounds))]

        # Decrease the mutation rate
        mutation_rate = min(mutation_rate, 0.1)

        # Check if the mutation rate is zero
        if mutation_rate == 0:
            break

    return best_individual, best_fitness

# Example usage:
def test_func(x):
    return np.sum(x**2)

best_individual, best_fitness = differential_evolution(test_func, [-5.0, 5.0], 0.1, 100)
print("Best solution:", best_individual)
print("Best fitness:", best_fitness)

# Genetic Algorithm
def genetic_algorithm(func, bounds, mutation_rate, population_size):
    # Initialize the population
    population = [np.random.uniform(bounds[0], bounds[1], size=(population_size, len(bounds))) for _ in range(population_size)]

    # Initialize the mutation rate
    mutation_rate = mutation_rate / population_size

    # Evaluate the function at the initial population
    fitness = [func(individual) for individual in population]

    # Initialize the best solution and its fitness
    best_individual = population[fitness.index(max(fitness))]
    best_fitness = max(fitness)

    # Initialize the mutation counter
    mutation_counter = 0

    while mutation_counter < population_size:
        # Select the next individual using genetic algorithm
        while True:
            # Select the next individual
            individual = population[np.random.choice(population_size)]

            # Evaluate the function at the selected individual
            fitness = [func(individual[i]) for i in range(len(bounds))]

            # Generate a new solution using genetic algorithm
            new_individual = individual + np.random.normal(0, 1, len(bounds)) * np.sqrt(max(fitness) / population_size)
            new_individual = np.clip(new_individual, bounds[0], bounds[1])

            # Evaluate the new solution
            new_fitness = [func(new_individual[i]) for i in range(len(bounds))]

            # Check if the new solution is better
            if new_fitness.index(max(new_fitness)) > fitness.index(max(fitness)):
                # Update the best solution and its fitness
                best_individual = new_individual
                best_fitness = max(new_fitness)

                # Update the mutation counter
                mutation_counter += 1

                # Check if the mutation rate is zero
                if mutation_rate == 0:
                    break

            # Update the mutation rate
            mutation_rate = min(mutation_rate, 0.1)

        # Update the population
        population = [individual if i not in [best_individual[i] for i in range(len(bounds))] else best_individual[i] for i in range(len(bounds))]

    return best_individual, best_fitness

# Example usage:
def test_func(x):
    return np.sum(x**2)

best_individual, best_fitness = genetic_algorithm(test_func, [-5.0, 5.0], 0.1, 100)
print("Best solution:", best_individual)
print("Best fitness:", best_fitness)