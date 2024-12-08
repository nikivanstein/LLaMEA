import numpy as np
from pyevolve import BaseOptimizer, Real, Minimize, Stats, mutation, selection, operator
from operator import itemgetter

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
        self.run(self.budget, func)

# Refine the strategy by changing individual lines with a probability of 0.1
class RefineNovelMutationOperator(BaseOptimizer):
    def __init__(self, budget, dim):
        super().__init__()
        self.budget = budget
        self.dim = dim
        self.stats = Stats()
        self.real = Real(['x'], [-5.0, 5.0], self.dim)
        self.refine_prob = 0.1

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        self.mutator = mutation.MutationOperator(mutation.GaussianMutation, mu=0.1, sigma=0.1)
        self.selector = selection.Selector(selection.TournamentSelector, tourn_size=5)
        self.operator = operator.Operators(['mutate','select'])
        self.refine = False
        for _ in range(self.budget):
            if np.random.rand() < self.refine_prob:
                self.refine = True
            else:
                self.refine = False
            if self.refine:
                for i in range(len(self.pop)):
                    if np.random.rand() < 0.1:
                        self.pop[i] = np.random.uniform(-5.0, 5.0)
            self.run(1, func)

# Usage
if __name__ == "__main__":
    optimizer = RefineNovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best().fitness}")