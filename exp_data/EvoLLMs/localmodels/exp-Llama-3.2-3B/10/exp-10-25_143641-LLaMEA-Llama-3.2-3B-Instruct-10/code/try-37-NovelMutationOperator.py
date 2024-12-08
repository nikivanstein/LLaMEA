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
        self.run(self.budget, func)

    def mutate(self, individual):
        for i in range(self.dim):
            if random.random() < self.probability:
                individual[i] += np.random.normal(0, 0.1)
                individual[i] = np.clip(individual[i], -5.0, 5.0)
        return individual

    def select(self, population):
        new_population = []
        for _ in range(len(population)):
            tournament = random.sample(population, 5)
            winner = max(tournament, key=lambda x: x.fitness)
            new_population.append(winner)
        return new_population

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best().f}")