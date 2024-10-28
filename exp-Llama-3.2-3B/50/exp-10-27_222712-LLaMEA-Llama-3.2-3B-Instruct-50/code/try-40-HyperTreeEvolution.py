import numpy as np
import random

class HyperTreeEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.tree = self.initialize_tree()
        self.population_size = 1  # Initialize population size to 1

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
                # Update the population
                self.population_size = 1
                self.population_size = np.random.choice([1, 2], p=[0.5, 0.5])
                if self.population_size == 1:
                    self.tree = self.initialize_tree()
                else:
                    # Randomly select a parent from the current population
                    parent = np.random.choice([self.tree], p=[1 / self.population_size] * self.population_size)
                    # Create a new child by perturbing the parent
                    child = parent.copy()
                    for i in range(self.dim):
                        child[i] += random.uniform(-0.1, 0.1)
                        child[i] = max(-5.0, min(5.0, child[i]))
                    # Evaluate the new child
                    value = func(tuple(child.values()))
                    # Update the population if the new child is better
                    if value < func(tuple(parent.values())):
                        self.tree = child
        return func(tuple(self.tree.values()))

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
evolution = HyperTreeEvolution(budget, dim)
best_value = evolution(func)
print(f"Best value: {best_value}")