import numpy as np
from pyevolve import BaseOptimizer, Real, Minimize, Stats
from pyevolve import mutation, selection, operator
from scipy.optimize import differential_evolution

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

        # Refine strategy with probability 0.1
        for _ in range(int(self.budget * 0.1)):
            ind = np.random.choice(self.pop, 1)[0]
            ind = self.mutator.mutate(ind)
            self.selector.select(ind, self.pop)

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = lambda x: x**2 + i
        res = optimizer(func)
        print(f"Function {i+1} optimized: {res.x[0]}")
        print(f"Fitness: {res.fx}")