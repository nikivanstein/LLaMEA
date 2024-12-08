import numpy as np
from pyevolution import Base, Core, Fitness, Operators, Selection, Crossover, Mutation
from pyevolution.algorithms import DifferentialEvolution

class NovelizedPyevolution(Base):
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.fitness = Fitness('Minimize', self.budget, self.dim)
        self.selection = Selection('Tournament', self.population_size, self.fitness)
        self.crossover = Crossover('SinglePoint', self.dim)
        self.mutation = Mutation('BitFlip', self.dim)
        self.operator = Operators.Operators(self.budget, self.dim)
        self.algorithm = DifferentialEvolution(self.fitness, self.selection, self.crossover, self.mutation, self.operator)

    def __call__(self, func):
        self.algorithm.evolve(generations=100)
        return self.algorithm.bestIndividual().fitness

# Usage:
novelized = NovelizedPyevolution(budget=100, dim=10)
result = novelized(func)
print(result)