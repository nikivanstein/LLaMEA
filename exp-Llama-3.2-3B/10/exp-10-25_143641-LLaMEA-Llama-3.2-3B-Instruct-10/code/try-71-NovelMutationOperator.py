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
        self.probability = 0.1

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        for _ in range(self.budget):
            # Apply probability to mutate or not
            if np.random.rand() < self.probability:
                self.mutator.mutate(self.pop)
            self.selector.select(self.pop)
            self.operator.apply(self.pop)
            if self.selector.fitness(self.pop)!= -1:
                self.pop.fitness.values[0] = func(self.pop.x)
                if self.pop.fitness.values[0] < self.pop.best fitness.values[0]:
                    self.pop.best = self.pop
        self.run(self.budget, func)

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best().fitness.values[0]}")