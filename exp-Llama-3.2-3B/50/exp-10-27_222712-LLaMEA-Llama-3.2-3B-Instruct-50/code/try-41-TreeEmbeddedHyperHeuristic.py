import numpy as np
import random

class TreeEmbeddedHyperHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.tree = self.initialize_tree()
        self.population_size = 1

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
                new_tree[dim] += np.random.uniform(-0.1, 0.1)
                new_tree[dim] = max(-5.0, min(5.0, new_tree[dim]))
            # Evaluate the new tree
            value = func(tuple(new_tree.values()))
            # Update the tree if the new value is better
            if value < func(tuple(self.tree.values())):
                self.tree = new_tree
                self.population_size += 1
                if self.population_size > 100:
                    # Select a random individual from the population
                    self.tree = random.choice(list(self.population()))
        return func(tuple(self.tree.values()))

    def population(self):
        # Initialize the population with the current tree
        population = [self.tree]
        for _ in range(99):
            # Select a random individual from the population
            parent = random.choice(population)
            # Create a child by randomly perturbing the parent
            child = parent.copy()
            for dim in range(self.dim):
                # Randomly perturb the selected dimension
                child[dim] += np.random.uniform(-0.1, 0.1)
                child[dim] = max(-5.0, min(5.0, child[dim]))
            # Evaluate the child
            value = func(tuple(child.values()))
            # Update the population if the child is better
            if value < func(tuple(parent.values())):
                population.append(child)
        return population

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
heuristic = TreeEmbeddedHyperHeuristic(budget, dim)
best_value = heuristic(func)
print(f"Best value: {best_value}")