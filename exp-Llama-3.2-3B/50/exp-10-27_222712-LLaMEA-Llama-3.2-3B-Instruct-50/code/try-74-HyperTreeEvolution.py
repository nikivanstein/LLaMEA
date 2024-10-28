import numpy as np
import random
import operator

class HyperTreeEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.tree = self.initialize_tree()

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
                perturbation = random.uniform(-0.1, 0.1)
                new_tree[dim] += perturbation
                new_tree[dim] = max(-5.0, min(5.0, new_tree[dim]))
            # Evaluate the new tree
            value = func(tuple(new_tree.values()))
            # Update the tree if the new value is better
            if value < func(tuple(self.tree.values())):
                self.tree = new_tree
                # Apply probability 0.5 to change individual lines of the tree
                for dim in range(self.dim):
                    if random.random() < 0.5:
                        new_tree[dim] += random.uniform(-0.05, 0.05)
                        new_tree[dim] = max(-5.0, min(5.0, new_tree[dim]))
            else:
                # If the new value is not better, revert the tree to its previous state
                for dim in range(self.dim):
                    new_tree[dim] -= random.uniform(-0.05, 0.05)
                    new_tree[dim] = max(-5.0, min(5.0, new_tree[dim]))
        return func(tuple(self.tree.values()))

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
evolution = HyperTreeEvolution(budget, dim)
best_value = evolution(func)
print(f"Best value: {best_value}")