import numpy as np
from scipy.optimize import minimize
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function

    def __call__(self, func, initial_guess, iterations):
        for _ in range(iterations):
            if _ >= self.budget:
                break
            best_x = initial_guess
            best_value = self.func(best_x)
            for i in range(self.dim):
                new_x = [x + random.uniform(-0.01, 0.01) for x in best_x]
                new_value = self.func(new_x)
                if new_value < best_value:
                    best_x = new_x
                    best_value = new_value
            initial_guess = best_x
        return best_x, best_value

    def novel_search(self, func, initial_guess, iterations, budget, probability):
        # Initialize the population with random individuals
        population = [initial_guess] * self.budget

        # Evaluate the fitness of each individual
        for _ in range(budget):
            fitness = [func(individual) for individual in population]
            # Select the fittest individuals based on probability
            selected_individuals = random.choices(population, weights=[fitness[i] / sum(fitness) for i in range(len(fitness))], k=probability * self.budget)
            # Replace the least fit individuals with the selected ones
            population = [individual for individual in population if individual not in selected_individuals] + selected_individuals

        # Evolve the population for the specified number of iterations
        for _ in range(iterations):
            for individual in population:
                if _ >= self.budget:
                    break
                # Evaluate the fitness of the individual
                fitness = [func(individual) for individual in population]
                # Refine the individual's strategy based on the fitness
                if random.random() < probability:
                    # Apply a small perturbation to the individual's strategy
                    new_strategy = [x + random.uniform(-0.01, 0.01) for x in individual]
                    # Evaluate the fitness of the new strategy
                    new_fitness = [func(new_strategy) for new_strategy in population]
                    # Select the fittest new strategy based on probability
                    selected_strategy = random.choices(population, weights=new_fitness, k=probability * self.budget)
                    # Replace the least fit strategy with the selected one
                    population = [individual for individual in population if individual not in selected_strategy] + selected_strategy

        # Return the fittest individual
        return population[0]

# Example usage:
budget = 10
dim = 2
optimizer = BlackBoxOptimizer(budget, dim)
best_individual, best_fitness = optimizer.novel_search(BlackBoxOptimizer.func, [-5.0, -5.0], 100, budget, 0.45)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)