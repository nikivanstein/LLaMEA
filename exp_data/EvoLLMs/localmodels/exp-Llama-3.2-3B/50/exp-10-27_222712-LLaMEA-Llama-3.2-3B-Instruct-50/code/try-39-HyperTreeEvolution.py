import numpy as np
import random
import operator

class HyperTreeEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.tree = self.initialize_tree()
        self.population = self.initialize_population()

    def initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = self.search_space[i]
        return tree

    def initialize_population(self):
        population = []
        for _ in range(self.budget // 2):
            new_tree = self.initialize_tree()
            population.append(new_tree)
        return population

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
                # Replace the worst individual in the population with the new tree
                self.population.sort(key=operator.itemgetter(np.sum), reverse=True)
                self.population[0] = new_tree
        return func(tuple(self.tree.values()))

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
evolution = HyperTreeEvolution(budget, dim)
best_value = evolution(func)
print(f"Best value: {best_value}")