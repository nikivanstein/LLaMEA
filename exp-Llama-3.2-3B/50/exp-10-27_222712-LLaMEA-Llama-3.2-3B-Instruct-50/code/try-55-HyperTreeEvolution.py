import numpy as np
import random
from deap import base, creator, tools, algorithms

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
                new_tree[dim] += random.uniform(-0.1, 0.1)
                new_tree[dim] = max(-5.0, min(5.0, new_tree[dim]))
            # Evaluate the new tree
            value = func(tuple(new_tree.values()))
            # Update the tree if the new value is better
            if value < func(tuple(self.tree.values())):
                self.tree = new_tree
        return func(tuple(self.tree.values()))

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
evolution = HyperTreeEvolution(budget, dim)
best_value = evolution(func)
print(f"Best value: {best_value}")

# Modified HyperTreeEvolution to use DEAP library
class HyperTreeEvolution_DEAP(HyperTreeEvolution):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population = self.initialize_population()
        self.logger = tools.Logbook()

    def initialize_population(self):
        # Initialize the population with random trees
        population = []
        for _ in range(self.budget):
            tree = {}
            for i in range(self.dim):
                tree[i] = self.search_space[i]
            population.append(toolbox Individual(tree))
        return population

    def evaluate_fitness(self, individual):
        # Evaluate the fitness of an individual
        value = func(tuple(individual.values()))
        self.logger.record("fitness", value)
        return value

    def select_parents(self, population):
        # Select parents for crossover
        parents = tools.selTournament(population, k=2)
        return parents

    def crossover(self, parents):
        # Perform crossover
        offspring = []
        for _ in range(self.budget):
            parent1, parent2 = self.select_parents(population)
            child = toolbox.cxSimulatedBinaryBoundedScaling(parent1, parent2)
            offspring.append(child)
        return offspring

    def mutate(self, offspring):
        # Perform mutation
        for child in offspring:
            # Randomly perturb the selected dimension
            dim = random.randint(0, self.dim-1)
            child.values[dim] += random.uniform(-0.1, 0.1)
            child.values[dim] = max(-5.0, min(5.0, child.values[dim]))
        return offspring

    def run(self):
        # Run the evolution
        for _ in range(self.budget):
            offspring = self.crossover(self.select_parents(self.population))
            offspring = self.mutate(offspring)
            self.population = offspring
            # Evaluate the fitness of the new population
            for individual in self.population:
                self.evaluate_fitness(individual)
            # Update the tree if the best individual is better
            if self.evaluate_fitness(self.population[0]) < self.evaluate_fitness(self.tree):
                self.tree = self.population[0]

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
evolution = HyperTreeEvolution_DEAP(budget, dim)
evolution.run()
best_value = evolution.tree.evaluate(func)
print(f"Best value: {best_value}")