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
        self.p = 0.1  # mutation probability

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        self.mutator = mutation.MutationOperator(mutation.GaussianMutation, mu=0.1, sigma=0.1)
        self.selector = selection.Selector(selection.TournamentSelector, tourn_size=5)
        self.operator = operator.Operators(['mutate','select'])
        self.run(self.budget, func)

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best().fitness.values[0]}")

# Refine the strategy by changing individual lines with probability 0.1
def refine_strategy(optimizer):
    for i in range(len(optimizer.pop)):
        if np.random.rand() < 0.1:
            for j in range(len(optimizer.real.individuals[i])):
                optimizer.real.individuals[i][j] = np.random.uniform(-5.0, 5.0)
    return optimizer

# Refine the strategy 10 times
for _ in range(10):
    optimizer = refine_strategy(optimizer)