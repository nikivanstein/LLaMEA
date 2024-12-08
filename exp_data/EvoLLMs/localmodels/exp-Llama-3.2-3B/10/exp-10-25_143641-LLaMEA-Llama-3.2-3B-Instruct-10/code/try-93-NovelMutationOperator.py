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

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        self.mutator = mutation.MutationOperator(mutation.GaussianMutation, mu=0.1, sigma=0.1)
        self.selector = selection.Selector(selection.TournamentSelector, tourn_size=5)
        self.operator = operator.Operators(['mutate','select'])
        self.line_search = 0.1
        self.line_search_prob = 0.1
        for _ in range(self.budget):
            self.run(func)
            if np.random.rand() < self.line_search_prob:
                for i in range(len(self.pop)):
                    if self.pop[i].fitness.value < self.pop[i].fitness.best:
                        new_individual = self.mutator.mutate(self.pop[i])
                        self.pop[i] = self.selector.select([self.pop[i], new_individual])
                        if self.pop[i].fitness.value < self.pop[i].fitness.best:
                            self.pop[i] = self.selector.select([self.pop[i], self.real.create()])
        self.best_individual = self.pop[0]

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best().fitness.value}")