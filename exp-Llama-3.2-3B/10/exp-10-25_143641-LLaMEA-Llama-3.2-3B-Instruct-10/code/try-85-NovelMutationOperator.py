import numpy as np
from pyevolve import BaseOptimizer, Real, Minimize, Stats
from pyevolve import mutation, selection, operator
import random

class NovelMutationOperator(BaseOptimizer):
    def __init__(self, budget, dim):
        super().__init__()
        self.budget = budget
        self.dim = dim
        self.stats = Stats()
        self.real = Real(['x'], [-5.0, 5.0], self.dim)

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        self.mutator = mutation.MutationOperator(mutation.GaussianMutation, mu=0.1, sigma=0.1)
        self.selector = selection.Selector(selection.TournamentSelector, tourn_size=5)
        self.operator = operator.Operators(['mutate','select'])
        self.probability = 0.1

        for _ in range(self.budget):
            # Select 5 individuals with tournament selection
            selected = self.selector.select(self.pop, self.pop, tourn_size=5)
            # Mutate the selected individuals with probability 0.1
            mutated = [self.mutator.mutate(individual) if random.random() < self.probability else individual for individual in selected]
            # Replace the selected individuals with the mutated ones
            self.pop = mutated
            # Evaluate the fitness of the population
            self.stats.clear()
            self.stats.register(self.pop)

        # Return the best individual
        return self.pop[0]

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimized_func = optimizer(func)
        print(f"Function {i+1} optimized: {optimized_func}")