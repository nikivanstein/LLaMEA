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
        self.prob_change = 0.1  # 10% probability of changing individual lines
        for _ in range(self.budget):
            if np.random.rand() < self.prob_change:
                for i in range(self.dim):
                    self.real.individuals[i].x[i] = np.random.uniform(-5.0, 5.0)
            self.mutator.mutate(self.pop)
            self.selector.select(self.pop)
            self.operator.operate(self.pop)
            self.eval_func(func, self.pop)

    def eval_func(self, func, individual):
        return func(individual.x)

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best().fitness.values}')