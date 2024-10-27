import numpy as np
import random

class EvolutionaryTreeBasedOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.tree_size = 10
        self.warmup = 10
        self.c1 = 1.5
        self.c2 = 1.5
        self.rho = 0.99
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = self.initialize_population()
        self.tree = self.initialize_tree()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            population.append(individual)
        return population

    def initialize_tree(self):
        tree = []
        for _ in range(self.tree_size):
            tree.append(np.random.uniform(self.lower_bound, self.upper_bound, self.dim))
        return tree

    def evaluate(self, func):
        for individual in self.population:
            func(individual)

    def update(self):
        for _ in range(self.tree_size):
            for individual in self.population:
                r1 = np.random.uniform(0, 1)
                r2 = np.random.uniform(0, 1)
                v1 = r1 * self.c1 * (individual - self.tree[random.randint(0, self.tree_size - 1)])
                v2 = r2 * self.c2 * (self.tree[random.randint(0, self.tree_size - 1)] - individual)
                individual += v1 + v2
                if individual[0] < self.lower_bound:
                    individual[0] = self.lower_bound
                if individual[0] > self.upper_bound:
                    individual[0] = self.upper_bound
                if individual[1] < self.lower_bound:
                    individual[1] = self.lower_bound
                if individual[1] > self.upper_bound:
                    individual[1] = self.upper_bound
                if random.random() < 0.2:
                    individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def mutation(self):
        mutation_rate = 0.25
        for individual in self.population:
            if random.random() < mutation_rate:
                index = np.random.randint(0, self.dim)
                individual[index] = self.tree[np.random.randint(0, self.tree_size)]

    def diversity(self):
        diversity = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            for j in range(self.population_size):
                distance = np.linalg.norm(self.population[i] - self.population[j])
                diversity[i, :] = np.append(distance, 1 - distance)
        return diversity

    def adaptive(self):
        diversity = self.diversity()
        for i in range(self.population_size):
            min_diversity = np.inf
            min_index = -1
            for j in range(self.population_size):
                if diversity[j, 0] < min_diversity:
                    min_diversity = diversity[j, 0]
                    min_index = j
            if diversity[min_index, 0] < 0.5:
                self.population[i] = self.population[min_index]

    def run(self, func):
        for _ in range(self.warmup):
            self.evaluate(func)
            self.update()
            self.adaptive()
            self.mutation()
        for _ in range(self.budget - self.warmup):
            self.evaluate(func)
            self.update()
            self.adaptive()
            self.mutation()
        return self.population[np.argmin([func(individual) for individual in self.population])]

# Example usage:
def func(x):
    return np.sum(x**2)

evolutionary_tree_based_optimization = EvolutionaryTreeBasedOptimization(100, 10)
result = evolutionary_tree_based_optimization(func)
print(result)