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
                # Add the new tree to the population
                self.population.append(self.tree)
            else:
                # With probability 0.5, replace the current tree with a new one
                if random.random() < self.probability:
                    self.tree = self.initialize_tree()
                    self.population.append(self.tree)
        # Return the best individual in the population
        return self.population[0]

    def evaluate_population(self, func):
        # Evaluate each individual in the population
        values = []
        for individual in self.population:
            value = func(tuple(individual.values()))
            values.append(value)
        # Return the best individual and its value
        best_individual = self.population[values.index(min(values))]
        return best_individual, min(values)

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
evolution = HyperTreeEvolution(budget, dim)
best_individual, best_value = evolution(func)
print(f"Best individual: {best_individual}")
print(f"Best value: {best_value}")