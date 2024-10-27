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
        self.refine_prob = 0.1

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        self.mutator = mutation.MutationOperator(mutation.GaussianMutation, mu=0.1, sigma=0.1)
        self.selector = selection.Selector(selection.TournamentSelector, tourn_size=5)
        self.operator = operator.Operators(['mutate','select'])
        self.refine = False

        for _ in range(self.budget):
            if random.random() < self.refine_prob:
                self.refine = True
            self.run(func)

            if self.refine:
                new_individual = self.pop[np.random.randint(len(self.pop))]
                new_individual = [x + random.uniform(-0.1, 0.1) for x in new_individual]
                self.pop[np.random.randint(len(self.pop))] = new_individual

        self.best = self.pop[np.argmin([func(x) for x in self.pop])]

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best()}")