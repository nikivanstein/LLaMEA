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
        # Initialize population
        population = [initial_guess.copy() for _ in range(100)]

        for _ in range(iterations):
            # Evaluate fitness of each individual
            fitnesses = [self.func(individual) for individual in population]
            population = [individual for _, individual in sorted(zip(fitnesses, population), reverse=True)]

            # Select parents
            parents = population[:int(0.45 * len(population))]

            # Create offspring
            offspring = []
            while len(offspring) < len(parents):
                parent1, parent2 = random.sample(parents, 2)
                child = (parent1 + 2 * parent2) / 3
                if np.random.rand() < 0.5:
                    child[0] += random.uniform(-0.01, 0.01)
                offspring.append(child)

            # Mutate offspring
            mutated_offspring = [copy.deepcopy(offspring[i]) for i in range(len(offspring))]
            for individual in mutated_offspring:
                for j in range(self.dim):
                    if random.random() < 0.05:
                        individual[j] += random.uniform(-0.01, 0.01)

            # Replace worst individuals with offspring
            population = mutated_offspring[:int(0.45 * len(population))]

        # Select best individual
        best_individual = population[int(0.45 * len(population)) - 1]

        # Optimize best individual
        best_individual, best_value = self.optimize_best_individual(best_individual, func, self.budget)

        return best_individual, best_value

    def optimize_best_individual(self, individual, func, budget):
        for _ in range(budget):
            if _ >= len(individual):
                break
            best_value = func(individual)
            for i in range(self.dim):
                new_individual = individual.copy()
                new_individual[i] += random.uniform(-0.01, 0.01)
                new_value = func(new_individual)
                if new_value < best_value:
                    best_individual = new_individual
                    best_value = new_value
        return best_individual, best_value

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 