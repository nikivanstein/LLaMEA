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
        self.p = 0.1
        self.line_search = True

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        self.mutator = mutation.MutationOperator(mutation.GaussianMutation, mu=0.1, sigma=0.1)
        self.selector = selection.Selector(selection.TournamentSelector, tourn_size=5)
        self.operator = operator.Operators(['mutate','select'])
        self.run(self.budget, func)

        if self.line_search:
            for i in range(self.budget):
                best_individual = self.best()
                if best_individual is not None:
                    best_x = best_individual['x']
                    best_fx = func(best_x)
                    new_x = best_x + random.uniform(-0.1, 0.1)
                    new_fx = func(new_x)
                    if new_fx < best_fx:
                        self.pop[i]['x'] = new_x

    def mutate(self, individual):
        if random.random() < self.p:
            mutation_index = random.randint(0, self.dim-1)
            mutation_value = random.uniform(-0.1, 0.1)
            individual['x'][mutation_index] += mutation_value
            if individual['x'][mutation_index] < -5.0:
                individual['x'][mutation_index] = -5.0
            elif individual['x'][mutation_index] > 5.0:
                individual['x'][mutation_index] = 5.0
        return individual

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best()}")