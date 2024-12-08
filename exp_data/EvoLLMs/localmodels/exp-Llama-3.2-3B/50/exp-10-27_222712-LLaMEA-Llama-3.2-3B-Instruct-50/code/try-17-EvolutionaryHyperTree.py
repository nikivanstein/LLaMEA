import numpy as np
import random

class EvolutionaryHyperTree:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.tree = self.initialize_tree()
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5

    def initialize_tree(self):
        tree = {}
        for i in range(self.dim):
            tree[i] = self.search_space[i]
        return tree

    def __call__(self, func):
        population = [self.tree.copy() for _ in range(self.population_size)]
        for _ in range(self.budget):
            # Select parents for crossover
            parents = random.sample(population, 2)
            parent1, parent2 = parents
            # Perform crossover
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)
            # Perform mutation
            child1 = self.mutate(child1, self.mutation_rate)
            child2 = self.mutate(child2, self.mutation_rate)
            # Evaluate the children
            value1 = func(tuple(child1.values()))
            value2 = func(tuple(child2.values()))
            # Replace the worst individual
            population[np.argmin([func(tuple(individual.values())) for individual in population])] = child1
            population[np.argmin([func(tuple(individual.values())) for individual in population])] = child2
        # Return the best individual
        return func(tuple(max(population, key=lambda individual: func(tuple(individual.values()))).values()))

    def crossover(self, parent1, parent2):
        child = parent1.copy()
        for i in range(self.dim):
            if random.random() < self.crossover_rate:
                child[i] = random.choice([parent1[i], parent2[i]])
        return child

    def mutate(self, individual, mutation_rate):
        for i in range(self.dim):
            if random.random() < mutation_rate:
                individual[i] += random.uniform(-0.1, 0.1)
                individual[i] = max(-5.0, min(5.0, individual[i]))
        return individual

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
evolution = EvolutionaryHyperTree(budget, dim)
best_value = evolution(func)
print(f"Best value: {best_value}")