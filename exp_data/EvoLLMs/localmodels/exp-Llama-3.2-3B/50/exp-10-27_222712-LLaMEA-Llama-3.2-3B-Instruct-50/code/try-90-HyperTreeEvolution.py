import numpy as np
import random
import operator

class HyperTreeEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.tree = self.initialize_tree()
        self.population = [self.tree]

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
            # Add the new tree to the population
            self.population.append(new_tree)

        # Select the best individual from the population
        self.tree = max(self.population, key=operator.attrgetter('fitness'))

    def evaluate_population(self, func):
        # Evaluate the fitness of each individual in the population
        for individual in self.population:
            value = func(tuple(individual.values()))
            individual['fitness'] = value

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
evolution = HyperTreeEvolution(budget, dim)
evolution.evaluate_population(func)
best_value = evolution.tree['fitness']
print(f"Best value: {best_value}")