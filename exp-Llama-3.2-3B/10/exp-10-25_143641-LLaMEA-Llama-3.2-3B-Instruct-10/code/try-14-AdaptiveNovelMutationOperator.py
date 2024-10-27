import numpy as np
from pyevolve import BaseOptimizer, Real, Minimize, Stats
from pyevolve import mutation, selection, operator

class AdaptiveNovelMutationOperator(BaseOptimizer):
    def __init__(self, budget, dim):
        super().__init__()
        self.budget = budget
        self.dim = dim
        self.stats = Stats()
        self.real = Real(['x'], [-5.0, 5.0], self.dim)
        self.mutator = mutation.MutationOperator(mutation.GaussianMutation, mu=0.1, sigma=0.1)
        self.selector = selection.Selector(selection.TournamentSelector, tourn_size=5)
        self.operator = operator.Operators(['mutate','select'])
        self.adaptation_rate = 0.1

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        for _ in range(self.budget):
            self.selector.select(self.pop, self.real, self.mutator, self.operator)
            self.mutator.mutate(self.pop, self.real)
            if np.random.rand() < self.adaptation_rate:
                self.mutator.mu *= 0.9
                self.mutator.sigma *= 0.9
            if self.mutator.mu < 0.01:
                self.mutator.mu = 0.01
            if self.mutator.sigma < 0.01:
                self.mutator.sigma = 0.01
            if self.mutator.mu > 1.0:
                self.mutator.mu = 1.0
            if self.mutator.sigma > 1.0:
                self.mutator.sigma = 1.0
            self.pop = self.real.create()
        self.run(self.pop, func)

# Usage
if __name__ == "__main__":
    optimizer = AdaptiveNovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best())