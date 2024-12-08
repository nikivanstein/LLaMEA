import numpy as np
import random

class HyperTree:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.selected_solution = None

    def __call__(self, func):
        if self.selected_solution is not None:
            self.update_solution(func)
        else:
            self.population = [self.create_individual(func)]
            self.selected_solution = random.choice(self.population)

        for _ in range(self.budget - 1):
            self.population.append(self.evolve(self.population))

        self.selected_solution = max(self.population, key=lambda x: x.fitness)

    def create_individual(self, func):
        individual = {
            'fitness': func(np.array([-5.0] * self.dim)),
            'params': np.random.uniform(-5.0, 5.0, self.dim)
        }
        return individual

    def evolve(self, population):
        new_population = []
        for _ in range(len(population)):
            parent1, parent2 = random.sample(population, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        return new_population

    def crossover(self, parent1, parent2):
        child = {
            'fitness': 0,
            'params': np.zeros(self.dim)
        }
        for i in range(self.dim):
            if random.random() < 0.5:
                child['params'][i] = (parent1['params'][i] + parent2['params'][i]) / 2
            else:
                child['params'][i] = parent1['params'][i]
        child['fitness'] = func(child['params'])
        return child

    def mutate(self, individual):
        mutation_rate = 0.1
        for i in range(self.dim):
            if random.random() < mutation_rate:
                individual['params'][i] += random.uniform(-1.0, 1.0)
                individual['fitness'] = func(individual['params'])
        return individual

    def update_solution(self, func):
        probability = 0.3
        for individual in self.population:
            if random.random() < probability:
                individual['params'] = (individual['params'] + self.selected_solution['params']) / 2
                individual['fitness'] = func(individual['params'])
        self.selected_solution = max(self.population, key=lambda x: x.fitness)

# Example usage:
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
hyper_tree = HyperTree(budget, dim)
hyper_tree('func')