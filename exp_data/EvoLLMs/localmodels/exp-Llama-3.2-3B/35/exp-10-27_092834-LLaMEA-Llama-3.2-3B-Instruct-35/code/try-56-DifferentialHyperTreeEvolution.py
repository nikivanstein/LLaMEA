import random
import numpy as np

class DifferentialHyperTreeEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.tree = self._initialize_tree()
        self.differential_evolution_params = {'pop_size': 10,'mutation_probability': 0.35, 'crossover_probability': 0.35}
        self.fitness_history = []

    def _initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = {'lower': -5.0, 'upper': 5.0, 'value': random.uniform(-5.0, 5.0)}
        return tree

    def __call__(self, func):
        for _ in range(self.budget):
            self._evaluate_and_mutate_tree(func)

    def _evaluate_and_mutate_tree(self, func):
        fitness = func(self.tree)
        self.fitness_history.append(fitness)
        if fitness == 0:
            return  # termination condition

        # Differential Evolution
        population = [self.tree.copy() for _ in range(self.differential_evolution_params['pop_size'])]
        for i in range(self.differential_evolution_params['pop_size']):
            for j in range(self.differential_evolution_params['pop_size']):
                if i!= j:
                    parent1, parent2 = random.sample(population, 2)
                    child = {'value': (parent1['value'] + parent2['value']) / 2}
                    child['value'] = max(child['value'], parent1['lower'])
                    child['value'] = min(child['value'], parent1['upper'])
                    child['value'] = max(child['value'], parent2['lower'])
                    child['value'] = min(child['value'], parent2['upper'])
                    population[i] = child

        # Hyper-Tree Evolution
        if random.random() < self.differential_evolution_params['mutation_probability']:
            mutated_tree = self._mutate_tree(population[0])
            if random.random() < self.differential_evolution_params['crossover_probability']:
                mutated_tree = self._crossover_tree(mutated_tree, population[1])
            self.tree = mutated_tree

        # Crossover
        if random.random() < self.differential_evolution_params['crossover_probability']:
            other_tree = self._initialize_tree()
            for i in range(self.dim):
                if random.random() < 0.5:
                    other_tree[i]['value'] = self.tree[i]['value']
            self.tree = other_tree

    def _mutate_tree(self, tree):
        mutated_tree = tree.copy()
        for i in range(self.dim):
            if random.random() < 0.5:
                mutated_tree[i]['value'] += random.uniform(-1.0, 1.0)
                if mutated_tree[i]['value'] < mutated_tree[i]['lower']:
                    mutated_tree[i]['value'] = mutated_tree[i]['lower']
                elif mutated_tree[i]['value'] > mutated_tree[i]['upper']:
                    mutated_tree[i]['value'] = mutated_tree[i]['upper']
        return mutated_tree

    def _crossover_tree(self, tree1, tree2):
        mutated_tree = tree1.copy()
        for i in range(self.dim):
            if random.random() < 0.5:
                mutated_tree[i]['value'] = tree2[i]['value']
        return mutated_tree

    def get_tree(self):
        return self.tree

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

budget = 100
dim = 2
evolution = DifferentialHyperTreeEvolution(budget, dim)
evolution()
tree = evolution.get_tree()
print(tree)