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

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        for i in range(self.budget):
            if i < self.budget * 0.1:
                # Refine mutation operator
                self.mutator.mu = random.uniform(0.05, 0.15)
                self.mutator.sigma = random.uniform(0.05, 0.15)
            self.pop, self.best, self.log = self.run(self.pop, func, self.stats)
            # Randomly select an individual to mutate
            if random.random() < 0.1:
                self.mutator.mutate(self.pop, self.real)
        self.run(self.pop, func, self.stats)

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best().fitness.value}")