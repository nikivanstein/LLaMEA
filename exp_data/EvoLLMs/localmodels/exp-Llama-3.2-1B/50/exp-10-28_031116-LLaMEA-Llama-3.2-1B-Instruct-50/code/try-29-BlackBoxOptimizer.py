import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()
        self.population_dict = self.initialize_population_dict()
        self.fitness_dict = self.initialize_fitness_dict()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            dim = random.randint(-5.0, 5.0)
            func = lambda x: np.sum(np.abs(x))
            population.append(func)
        return population

    def initialize_population_dict(self):
        population_dict = {}
        for func in self.population:
            func_dict = {}
            for _ in range(self.dim):
                func_dict[_] = random.uniform(-5.0, 5.0)
            population_dict[func] = func_dict
        return population_dict

    def initialize_fitness_dict(self):
        fitness_dict = {}
        for func in self.population:
            fitness = 0
            for _ in range(self.dim):
                fitness += np.abs(func(random.uniform(-5.0, 5.0)))
            fitness_dict[func] = fitness
        return fitness_dict

    def __call__(self, func):
        if func not in self.population_dict:
            self.population.append(func)
            self.population_dict[func] = {}
            self.fitness_dict[func] = {}
        while len(self.population) > self.budget:
            func = random.choice(list(self.population_dict.keys()))
            if func in self.fitness_dict:
                fitness = self.fitness_dict[func]
                if fitness < self.population_dict[func][self.dim]:
                    self.population_dict[func][self.dim] = fitness
                    self.fitness_dict[func][self.dim] = fitness
                    self.population_dict[func].pop(self.dim)
                    self.fitness_dict[func].pop(self.dim)
            else:
                self.population_dict[func].pop(self.dim)
                self.fitness_dict[func].pop(self.dim)
            self.population_dict[func].pop(self.dim)
            self.fitness_dict[func].pop(self.dim)
        return self.population_dict[func][self.dim]

    def mutate(self, func):
        if func in self.population_dict:
            for _ in range(random.randint(0, 10)):
                dim = random.randint(-5.0, 5.0)
                func = func + dim
            self.population_dict[func].pop(dim)
            self.fitness_dict[func].pop(dim)
            self.population_dict[func].append(dim)
            self.fitness_dict[func].append(dim)
        return func

    def crossover(self, parent1, parent2):
        if parent1 in self.population_dict:
            for _ in range(random.randint(0, 10)):
                dim = random.randint(-5.0, 5.0)
                parent1 = parent1 + dim
            for _ in range(random.randint(0, 10)):
                dim = random.randint(-5.0, 5.0)
                parent2 = parent2 + dim
            self.population_dict[parent1].append(parent2)
            self.fitness_dict[parent1].append(parent2)
            self.population_dict[parent2].append(parent1)
            self.fitness_dict[parent2].append(parent1)
        return parent1, parent2