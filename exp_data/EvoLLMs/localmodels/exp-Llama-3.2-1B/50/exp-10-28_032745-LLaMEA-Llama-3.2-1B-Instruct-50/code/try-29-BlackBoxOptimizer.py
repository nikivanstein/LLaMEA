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

    def __call__(self, func, initial_guess, iterations):
        # Initialize population with random initial guesses
        population = [copy.deepcopy(initial_guess) for _ in range(100)]

        # Evaluate population fitness
        fitnesses = [self.func(individual) for individual in population]

        # Select fittest individuals
        fittest_individuals = sorted(zip(fitnesses, population), reverse=True)[:self.budget]

        # Initialize new population
        new_population = []
        for _ in range(iterations):
            # Select parents using tournament selection
            parents = []
            for _ in range(self.budget):
                parent1, parent2 = random.sample(fittest_individuals, 2)
                tournament = [p[0] for p in zip(parent1, parent2) if p[1] > parent1[1]]
                tournament.sort()
                tournament_parent = tournament[0]
                tournament_individual = copy.deepcopy(parent1)
                tournament_individual[1] = tournament_parent[1]
                parents.append(tournament_individual)

            # Crossover (recombination) offspring
            offspring = []
            for _ in range(self.dim):
                parent1, parent2 = random.sample(parents, 2)
                child = [random.uniform(self.search_space[0], self.search_space[1]) for _ in range(self.dim)]
                child[0] = parent1[0]
                child[1] = parent2[1]
                offspring.append(copy.deepcopy(child))

            # Mutate offspring
            for individual in offspring:
                for i in range(self.dim):
                    if random.random() < 0.1:
                        individual[i] += random.uniform(-0.01, 0.01)

            # Add mutated offspring to new population
            new_population.extend(offspring)

        # Replace old population with new population
        population = new_population

        # Evaluate new population fitness
        new_fitnesses = [self.func(individual) for individual in population]

        # Select fittest individuals
        new_fittest_individuals = sorted(zip(new_fitnesses, population), reverse=True)[:self.budget]

        # Return fittest individual and its fitness
        return new_fittest_individuals[0][0], new_fittest_individuals[0][1]

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 