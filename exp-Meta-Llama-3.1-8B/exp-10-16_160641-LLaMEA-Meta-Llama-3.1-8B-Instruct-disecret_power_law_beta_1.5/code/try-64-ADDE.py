import numpy as np
import random

class ADDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.population_size = 50
        self.num_parents = int(0.2 * self.population_size)
        self.num_offspring = self.population_size - self.num_parents
        self.diffusion_rate = 0.1
        self.driven_evolution_rate = 0.05
        self.probability_adjustment_rate = 0.1  # new parameter

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

    def probability_adjustment(self, population, func):
        fitness = self.evaluate(func, population)
        sorted_idx = np.argsort(fitness)
        top_idx = sorted_idx[:int(0.1 * self.population_size)]  # top 10%
        bottom_idx = sorted_idx[-int(0.1 * self.population_size):]  # bottom 10%
        for i in top_idx:
            self.diffusion_rate += self.probability_adjustment_rate
        for i in bottom_idx:
            self.diffusion_rate -= self.probability_adjustment_rate
        if self.diffusion_rate < 0:
            self.diffusion_rate = 0
        elif self.diffusion_rate > 1:
            self.diffusion_rate = 1

    def selection(self, population, func):
        fitness = self.evaluate(func, population)
        sorted_idx = np.argsort(fitness)
        return population[sorted_idx[:self.num_parents]]

    def __call__(self, func):
        population = self.initialize_population()
        for _ in range(self.budget):
            population = self.selection(population, func)
            population = self.probability_adjustment(population, func)
            population = self.diffusion(population)
            population = self.driven_evolution(population, func)
        return population[0]