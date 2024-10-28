import numpy as np
import random
import operator

class HyperTreeEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.tree = self.initialize_tree()
        self.population_size = 1
        self.probability = 0.5

    def initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = self.search_space[i]
        return tree

    def __call__(self, func):
        for _ in range(self.budget):
            # Select a subset of dimensions to vary
            subset = random.sample(range(self.dim), random.randint(1, self.dim))
            new_tree = self.tree.copy()
            for dim in subset:
                # Randomly perturb the selected dimension
                new_tree[dim] += random.uniform(-0.1, 0.1)
                new_tree[dim] = max(-5.0, min(5.0, new_tree[dim]))
            # Evaluate the new tree
            value = func(tuple(new_tree.values()))
            # Update the tree if the new value is better
            if value < func(tuple(self.tree.values())):
                self.tree = new_tree
        return func(tuple(self.tree.values()))

    def mutate_tree(self, tree):
        for dim in range(self.dim):
            if random.random() < self.probability:
                # Randomly change the value of the dimension
                tree[dim] += random.uniform(-0.1, 0.1)
                tree[dim] = max(-5.0, min(5.0, tree[dim]))
        return tree

    def crossover_tree(self, parent1, parent2):
        child = parent1.copy()
        for dim in range(self.dim):
            if random.random() < self.probability:
                # Randomly select a dimension from the parent2
                child[dim] = parent2[dim]
        return child

    def evolve(self, parent1, parent2):
        child = self.crossover_tree(parent1, parent2)
        child = self.mutate_tree(child)
        return child

    def evolve_population(self):
        population = [self.tree]
        for _ in range(self.population_size - 1):
            parent1, parent2 = random.sample(population, 2)
            child = self.evolve(parent1, parent2)
            population.append(child)
        return population

    def select_best(self, population):
        return max(population, key=operator.attrgetter('value'))

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
evolution = HyperTreeEvolution(budget, dim)
population = evolution.evolve_population()
best_individual = evolution.select_best(population)
best_value = func(tuple(best_individual.values()))
print(f"Best value: {best_value}")