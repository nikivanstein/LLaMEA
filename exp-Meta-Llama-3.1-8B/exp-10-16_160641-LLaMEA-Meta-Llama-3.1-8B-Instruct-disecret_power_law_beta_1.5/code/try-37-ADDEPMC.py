import numpy as np
import random

class ADDEPMC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.population_size = 50
        self.num_parents = int(0.2 * self.population_size)
        self.num_offspring = self.population_size - self.num_parents
        self.diffusion_rate = 0.1
        self.driven_evolution_rate = 0.05
        self.mutation_rate = 0.2
        self.crossover_rate = 0.3

    def initialize_population(self):
        return np.random.uniform(self.search_space[0], self.search_space[1], (self.population_size, self.dim))

    def evaluate(self, func, population):
        return np.array([func(ind) for ind in population])

    def diffusion(self, population):
        new_population = population.copy()
        for i in range(self.num_offspring):
            parent_idx = random.randint(0, self.population_size - 1)
            offspring = population[parent_idx].copy()
            for j in range(self.dim):
                if random.random() < self.diffusion_rate:
                    offspring[j] += random.uniform(-1, 1)
                    if offspring[j] < self.search_space[0]:
                        offspring[j] = self.search_space[0]
                    elif offspring[j] > self.search_space[1]:
                        offspring[j] = self.search_space[1]
            new_population[i] = offspring
        return new_population

    def driven_evolution(self, population, func):
        new_population = population.copy()
        for i in range(self.num_offspring):
            parent_idx = random.randint(0, self.population_size - 1)
            offspring = population[parent_idx].copy()
            for j in range(self.dim):
                if random.random() < self.driven_evolution_rate:
                    mutation = random.uniform(-1, 1)
                    if func(offspring + mutation * np.array([0] * self.dim)) < func(offspring):
                        offspring[j] += mutation
                        if offspring[j] < self.search_space[0]:
                            offspring[j] = self.search_space[0]
                        elif offspring[j] > self.search_space[1]:
                            offspring[j] = self.search_space[1]
            new_population[i] = offspring
        return new_population

    def mutation(self, population):
        new_population = population.copy()
        for i in range(self.num_offspring):
            parent_idx = random.randint(0, self.population_size - 1)
            offspring = population[parent_idx].copy()
            for j in range(self.dim):
                if random.random() < self.mutation_rate:
                    mutation = random.uniform(-1, 1)
                    offspring[j] += mutation
                    if offspring[j] < self.search_space[0]:
                        offspring[j] = self.search_space[0]
                    elif offspring[j] > self.search_space[1]:
                        offspring[j] = self.search_space[1]
            new_population[i] = offspring
        return new_population

    def crossover(self, population):
        new_population = population.copy()
        for i in range(self.num_offspring):
            parent_idx1 = random.randint(0, self.population_size - 1)
            parent_idx2 = random.randint(0, self.population_size - 1)
            offspring1 = population[parent_idx1].copy()
            offspring2 = population[parent_idx2].copy()
            crossover_point = random.randint(0, self.dim - 1)
            for j in range(crossover_point):
                offspring1[j], offspring2[j] = offspring2[j], offspring1[j]
            new_population[i] = offspring1
            new_population[self.population_size + i] = offspring2
        return new_population

    def selection(self, population, func):
        fitness = self.evaluate(func, population)
        sorted_idx = np.argsort(fitness)
        return population[sorted_idx[:self.num_parents]]

    def __call__(self, func):
        population = self.initialize_population()
        for _ in range(self.budget):
            population = self.selection(population, func)
            population = self.diffusion(population)
            population = self.driven_evolution(population, func)
            population = self.mutation(population)
            population = self.crossover(population)
        return population[0]