# Description: AdaptiveBlackBoxOptimizer: A novel metaheuristic algorithm for solving black box optimization problems
# Code: 
# ```python
import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.population = None
        self.population_history = None

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

        # Refine the population using genetic algorithm
        self.population = self.population_history[-1][1:] + [self.population_history[-1][0]]
        self.population_history.append((self.population, self.func_values))

        # Select the fittest individual
        fitnesses = [self.func_values[i] for i in range(len(self.func_values))]
        self.population = self.population[np.argsort(fitnesses)][::-1][:self.budget]

        # Evolve the population using adaptive mutation
        for _ in range(100):
            new_population = []
            for _ in range(self.budget):
                parent1, parent2 = random.sample(self.population, 2)
                child = (parent1 + parent2) / 2
                if random.random() < 0.35:
                    child = parent1 + random.uniform(-1, 1)
                new_population.append(child)
            self.population = new_population

# One-line description with the main idea
# AdaptiveBlackBoxOptimizer: A novel metaheuristic algorithm that uses a combination of adaptive mutation and genetic algorithm to optimize black box functions
# ```python