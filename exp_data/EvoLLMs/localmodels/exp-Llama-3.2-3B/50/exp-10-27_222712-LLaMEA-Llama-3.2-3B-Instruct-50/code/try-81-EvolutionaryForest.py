import numpy as np
import random

class EvolutionaryForest:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.forest = self.initialize_forest()

    def initialize_forest(self):
        forest = []
        for _ in range(100):  # Initialize with 100 trees
            tree = {}
            for i in range(self.dim):
                tree[i] = self.search_space[i]
            forest.append(tree)
        return forest

    def __call__(self, func):
        for _ in range(self.budget):
            # Select a random subset of dimensions to vary
            subset = random.sample(range(self.dim), random.randint(1, self.dim))
            new_trees = []
            for tree in self.forest:
                new_tree = tree.copy()
                for dim in subset:
                    # Randomly perturb the selected dimension
                    new_tree[dim] += random.uniform(-0.1, 0.1)
                    new_tree[dim] = max(-5.0, min(5.0, new_tree[dim]))
                new_trees.append(new_tree)
            # Evaluate the new trees
            values = [func(tuple(tree.values())) for tree in new_trees]
            # Update the forest if the new values are better
            if np.any(np.array(values) < np.array([func(tuple(tree.values())) for tree in self.forest])):
                self.forest = new_trees
        return func(tuple(self.forest[0].values()))

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
evolution = EvolutionaryForest(budget, dim)
best_value = evolution(func)
print(f"Best value: {best_value}")