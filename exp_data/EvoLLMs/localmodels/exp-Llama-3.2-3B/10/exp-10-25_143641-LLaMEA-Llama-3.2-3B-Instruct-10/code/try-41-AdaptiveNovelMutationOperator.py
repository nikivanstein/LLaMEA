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
        self.mutator_adaptation = 0.1  # adaptation rate for mutation operator
        self.selector_adaptation = 0.1  # adaptation rate for selector

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        for _ in range(self.budget):
            self.run(func)
            # adapt mutation operator
            if np.random.rand() < self.mutator_adaptation:
                self.mutator.mu *= 0.9
                self.mutator.sigma *= 0.9
            # adapt selector
            if np.random.rand() < self.selector_adaptation:
                self.selector.tourn_size *= 0.9
                if self.selector.tourn_size < 2:
                    self.selector.tourn_size = 2
            # adapt operator
            if np.random.rand() < self.mutator_adaptation:
                self.operator = operator.Operators(['mutate','select','crossover'])
                self.operator = operator.Operators(['select','crossover','mutate'])
            # adapt population size
            if np.random.rand() < self.selector_adaptation:
                self.pop = self.real.create()
                self.pop = self.pop[:len(self.pop)//2]
                self.pop = self.pop[:len(self.pop)//2]
            # select best individual
            self.best = self.real.select(self.pop, 1)[0]

# Usage
if __name__ == "__main__":
    optimizer = AdaptiveNovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best().fitness.values[0]}")