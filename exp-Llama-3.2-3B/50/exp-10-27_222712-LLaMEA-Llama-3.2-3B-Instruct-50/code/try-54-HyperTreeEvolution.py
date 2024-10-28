import numpy as np
import random

class HyperTreeEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.tree = self.initialize_tree()
        self.population_size = 1
        self.population = [self.tree]
        self.crossover_prob = 0.5

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
                if random.random() < self.crossover_prob:
                    new_tree[dim] += random.uniform(-0.1, 0.1)
                    new_tree[dim] = max(-5.0, min(5.0, new_tree[dim]))
            # Evaluate the new tree
            value = func(tuple(new_tree.values()))
            # Update the tree if the new value is better
            if value < func(tuple(self.tree.values())):
                self.tree = new_tree
            # Add the new tree to the population
            if len(self.population) < self.population_size:
                self.population.append(new_tree)
            # Perform crossover
            if len(self.population) > self.population_size:
                new_population = []
                while len(new_population) < self.population_size:
                    parent1 = random.choice(self.population)
                    parent2 = random.choice(self.population)
                    child = self.crossover(parent1, parent2)
                    new_population.append(child)
                self.population = new_population

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        for dim in range(self.dim):
            if random.random() < self.crossover_prob:
                child[dim] = (parent1[dim] + parent2[dim]) / 2
                child[dim] = max(-5.0, min(5.0, child[dim]))
        return child

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
evolution = HyperTreeEvolution(budget, dim)
best_value = evolution(func)
print(f"Best value: {best_value}")