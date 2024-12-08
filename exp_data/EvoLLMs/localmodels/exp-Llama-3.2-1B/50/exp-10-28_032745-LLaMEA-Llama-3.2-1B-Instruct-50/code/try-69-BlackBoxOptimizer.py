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
        # Initialize population with random individuals
        population = [initial_guess for _ in range(self.budget)]

        for _ in range(iterations):
            # Evaluate fitness of each individual in the population
            fitnesses = [self.func(individual) for individual in population]

            # Select parents using tournament selection
            parents = [population[np.random.choice(len(population))]] * 5  # Replace with 5% of the population
            for _ in range(5):
                # Select two parents using roulette wheel selection
                parent1 = random.choices(parents, weights=[fitness / sum(fitnesses) for fitness in fitnesses])[0]
                parent2 = random.choices(parents, weights=[fitness / sum(fitnesses) for fitness in fitnesses])[0]
                # Replace with 5% of the population
                parents = [population[np.random.choice(len(population))]] * 5 + [parent1, parent2]

            # Crossover (recombination) to generate offspring
            offspring = [self.func([x, y]) for x, y in zip(parent1, parent2)]

            # Mutate offspring to introduce genetic variation
            for _ in range(self.budget // 10):
                offspring[_ % len(offspring)] = [x + random.uniform(-0.01, 0.01) for x in offspring[_ % len(offspring)]]

            # Replace the least fit individual with the offspring
            population[np.argmin(fitnesses)] = offspring[0]

        return population[0], min(fitnesses)

# Novel metaheuristic algorithm for black box optimization using a novel search strategy
# 
# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 