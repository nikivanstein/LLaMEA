import numpy as np
from pyevolve import BaseOptimizer, Real, Minimize, Stats
from pyevolve import mutation, selection, operator

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
        self.refine_pop()
        self.run(self.budget, func)

    def refine_pop(self):
        for i in range(len(self.pop)):
            if np.random.rand() < 0.1:
                self.pop[i] = self.mutator.mutate(self.pop[i])
            if np.random.rand() < 0.1:
                self.selector.select(self.pop, k=1)
                self.pop[i] = self.pop[self.selector.tournament_size-1]

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best())