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
        self.change_probability = 0.1

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        for i in range(len(self.pop)):
            if random.random() < self.change_probability:
                index = random.randint(0, self.dim-1)
                self.pop[i][index] = random.uniform(-5.0, 5.0)
        self.mutator.set_params(mu=0.1, sigma=0.1)
        self.selector.set_params(tourn_size=5)
        self.operator.set_params(operators=['mutate','select'])
        self.run(self.budget, func)

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best())