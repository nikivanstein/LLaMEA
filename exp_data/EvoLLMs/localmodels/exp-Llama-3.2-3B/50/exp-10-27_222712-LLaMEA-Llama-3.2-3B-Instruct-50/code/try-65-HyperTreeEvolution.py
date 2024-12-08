import numpy as np
import random
import copy

class HyperTreeEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.tree = self.initialize_tree()
        self.population = [copy.deepcopy(self.tree) for _ in range(self.budget)]

    def initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = self.search_space[i]
        return tree

    def __call__(self, func):
        for _ in range(self.budget):
            # Select a subset of dimensions to vary
            subset = random.sample(range(self.dim), random.randint(1, self.dim))
            new_trees = []
            for tree in self.population:
                new_tree = copy.deepcopy(tree)
                for dim in subset:
                    # Randomly perturb the selected dimension
                    new_tree[dim] += random.uniform(-0.1, 0.1)
                    new_tree[dim] = max(-5.0, min(5.0, new_tree[dim]))
                new_trees.append(new_tree)
            # Evaluate the new trees
            values = [func(tuple(tree.values())) for tree in new_trees]
            # Update the population if the new values are better
            for i in range(len(values)):
                if values[i] < min(values):
                    self.population[i] = new_trees[i]
        return func(tuple(self.population[0].values()))

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
evolution = HyperTreeEvolution(budget, dim)
best_value = evolution(func)
print(f"Best value: {best_value}")