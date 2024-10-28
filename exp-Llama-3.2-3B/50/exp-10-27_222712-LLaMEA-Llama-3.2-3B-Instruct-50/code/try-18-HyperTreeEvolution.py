import numpy as np
import random

class HyperTreeEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.tree = self.initialize_tree()
        self.population_size = 100

    def initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = self.search_space[i]
        return tree

    def __call__(self, func):
        population = [self.tree.copy() for _ in range(self.population_size)]
        for _ in range(self.budget):
            # Select a subset of dimensions to vary
            subset = random.sample(range(self.dim), random.randint(1, self.dim))
            new_population = []
            for individual in population:
                new_individual = individual.copy()
                for dim in subset:
                    # Randomly perturb the selected dimension
                    new_individual[dim] += random.uniform(-0.1, 0.1)
                    new_individual[dim] = max(-5.0, min(5.0, new_individual[dim]))
                # Evaluate the new individual
                value = func(tuple(new_individual.values()))
                # Update the population if the new value is better
                if value < func(tuple(individual.values())):
                    new_population.append(new_individual)
                else:
                    new_population.append(individual)
            population = new_population
            # Select the best individual
            population.sort(key=lambda x: func(tuple(x.values())), reverse=True)
            self.tree = population[0]
        return func(tuple(self.tree.values()))

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
evolution = HyperTreeEvolution(budget, dim)
best_value = evolution(func)
print(f"Best value: {best_value}")