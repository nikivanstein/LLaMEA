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

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        self.mutator = mutation.MutationOperator(mutation.GaussianMutation, mu=0.1, sigma=0.1)
        self.selector = selection.Selector(selection.TournamentSelector, tourn_size=5)
        self.operator = operator.Operators(['mutate','select'])
        self.refine = True

        for _ in range(self.budget):
            if self.refine:
                # 10% chance to refine the individual
                if random.random() < 0.1:
                    # Randomly select an individual to refine
                    idx = random.randint(0, len(self.pop) - 1)
                    # Refine the individual by changing one of its lines
                    new_individual = list(self.pop[idx])
                    new_individual[random.randint(0, self.dim - 1)] = random.uniform(-5.0, 5.0)
                    self.pop[idx] = new_individual
            self.run(self.pop, func)

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best())