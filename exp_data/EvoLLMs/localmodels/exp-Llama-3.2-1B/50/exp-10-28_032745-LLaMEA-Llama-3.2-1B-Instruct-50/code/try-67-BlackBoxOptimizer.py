import numpy as np
from scipy.optimize import minimize
import random
import copy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function
        self.population_size = 100
        self.mutate_rate = 0.01

    def __call__(self, func, initial_guess, iterations):
        # Initialize the population with random initial guesses
        population = [copy.deepcopy(initial_guess) for _ in range(self.population_size)]

        for _ in range(iterations):
            # Evaluate the fitness of each individual in the population
            fitness = [self.func(individual) for individual in population]

            # Select the fittest individuals
            fittest_individuals = [individual for individual, fitness in zip(population, fitness) if fitness == max(fitness)]

            # Select parents using the tournament selection strategy
            parents = []
            for _ in range(self.population_size // 2):
                tournament_size = random.randint(2, self.population_size)
                tournament = random.sample(fittest_individuals, tournament_size)
                winner = max(tournament, key=self.func)
                parents.append(winner)

            # Mutate the parents
            mutated_parents = []
            for parent in parents:
                mutated_parent = copy.deepcopy(parent)
                for _ in range(self.mutate_rate * self.population_size):
                    if random.random() < 0.5:
                        mutated_parent[0] += random.uniform(-0.01, 0.01)
                        mutated_parent[1] += random.uniform(-0.01, 0.01)
                mutated_parents.append(mutated_parent)

            # Replace the fittest individuals with the mutated parents
            population = [mutated_parents[i] for i in range(self.population_size)]

        # Return the fittest individual
        best_individual, best_value = max(population, key=self.func)
        return best_individual, best_value

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy