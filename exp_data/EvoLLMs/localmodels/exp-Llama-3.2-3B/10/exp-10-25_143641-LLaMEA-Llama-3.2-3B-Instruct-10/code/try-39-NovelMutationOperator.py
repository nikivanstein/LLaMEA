import numpy as np
from pyevolve import BaseOptimizer, Real, Minimize, Stats, mutation, selection, operator
from functools import partial

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
        self.adaptation_prob = 0.1

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        for i in range(self.budget):
            if np.random.rand() < self.adaptation_prob:
                self.mutator.mutate(self.pop)
            self.selector.select(self.pop)
            self.operator.apply(self.pop)
            if self.pop.fitness.values[0] < func(self.pop.x):
                self.pop.fitness.values[0] = func(self.pop.x)
        self.run(self.budget, func)

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = partial(eval, 'lambda x: x**2 +'+ str(i))
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best().x}")