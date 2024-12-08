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
        self.mutator = mutation.MutationOperator(mutation.GaussianMutation, mu=0.1, sigma=0.1)
        self.selector = selection.Selector(selection.TournamentSelector, tourn_size=5)
        self.operator = operator.Operators(['mutate','select'])
        self.probability = 0.1

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        for i in range(self.budget):
            if random.random() < self.probability:
                individual = self.real.create()
                self.mutator.mutate(individual)
                self.selector.select(individual)
                self.operator.select(individual)
            else:
                individual = self.real.create()
                self.selector.select(individual)
                self.operator.select(individual)
            self.run(1, func)
        self.run(1, func)

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best())