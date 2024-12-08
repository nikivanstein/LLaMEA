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
        self.probability = 0.1

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        self.mutator = mutation.MutationOperator(mutation.GaussianMutation, mu=0.1, sigma=0.1)
        self.selector = selection.Selector(selection.TournamentSelector, tourn_size=5)
        self.operator = operator.Operators(['mutate','select'])
        
        # Perform probability-based line search
        for _ in range(self.budget):
            new_individual = self.real.create()
            new_individual = self.mutator.mutate(new_individual)
            new_individual = self.selector.select(new_individual, self.pop)
            new_individual = self.operator.apply(new_individual)
            
            # Check if the new individual is better than the current best
            if self.evaluate(func, new_individual) < self.evaluate(func, self.best):
                self.real.update(new_individual)
                self.best = new_individual
                
    def evaluate(self, func, individual):
        return func(individual)

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.real.best().x}")