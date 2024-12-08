import random
import numpy as np

class HyperTreeEvolutionDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.fitness_history = []
        self.population = [self.tree]

    def _initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
        return tree

    def __call__(self, func):
        for _ in range(self.budget):
            self._evaluate_and_mutate_tree(func)

    def _evaluate_and_mutate_tree(self, func):
        fitness = [func(individual) for individual in self.population]
        self.fitness_history.append(min(fitness))
        if min(fitness) == 0:
            return  # termination condition

        for i in range(len(self.population)):
            parent1, parent2 = random.sample(self.population, 2)
            child = self._crossover(parent1, parent2)
            self._mutate(child)
            self.population.append(child)

    def _crossover(self, parent1, parent2):
        child = self._initialize_tree()
        for i in range(self.dim):
            if random.random() < 0.5:  # crossover probability
                child[i]['value'] = (parent1[i]['value'] + parent2[i]['value']) / 2
                if child[i]['value'] < child[i]['lower']:
                    child[i]['value'] = child[i]['lower']
                elif child[i]['value'] > child[i]['upper']:
                    child[i]['value'] = child[i]['upper']
        return child

    def _mutate(self, individual):
        for i in range(self.dim):
            if random.random() < 0.5:  # mutation probability
                individual[i]['value'] += random.uniform(-1.0, 1.0)
                if individual[i]['value'] < individual[i]['lower']:
                    individual[i]['value'] = individual[i]['lower']
                elif individual[i]['value'] > individual[i]['upper']:
                    individual[i]['value'] = individual[i]['upper']

    def get_tree(self):
        return self.tree

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = HyperTreeEvolutionDE(budget, dim)
evolution()
tree = evolution.get_tree()
print(tree)