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
        for _ in range(self.budget):
            # 0.1 probability of changing individual lines
            if np.random.rand() < 0.1:
                self.real.individuals = self.mutator.mutate(self.real.individuals)
            self.selector.select(self.pop, self.real.individuals)
            self.operator.operator(self.pop, self.real.individuals)
            self.pop.fitness.values = self.real.fitness(self.pop)
            if self.real.fitness(self.pop).min < self.real.fitness(self.pop).max:
                self.real.individuals = self.pop
            self.pop = self.real.create()
        self.stats.clear()
        self.real.fitness(self.pop)
        self.stats.clear()
        self.stats.register(self.pop, self.real.fitness(self.pop))

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best().fitness.values}")