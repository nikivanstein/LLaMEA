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
        self.probability_change_individual = 1.0

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
            if random.random() < self.probability_change_individual:
                new_parent_idx = random.randint(0, self.population_size - 1)
                new_offspring = population[new_parent_idx].copy()
                for k in range(self.dim):
                    if random.random() < self.diffusion_rate:
                        new_offspring[k] += random.uniform(-1, 1)
                        if new_offspring[k] < self.search_space[0]:
                            new_offspring[k] = self.search_space[0]
                        elif new_offspring[k] > self.search_space[1]:
                            new_offspring[k] = self.search_space[1]
                new_population[i] = new_offspring
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
            if random.random() < self.probability_change_individual:
                new_parent_idx = random.randint(0, self.population_size - 1)
                new_offspring = population[new_parent_idx].copy()
                for k in range(self.dim):
                    if random.random() < self.driven_evolution_rate:
                        mutation = random.uniform(-1, 1)
                        if func(new_offspring + mutation * np.array([0] * self.dim)) < func(new_offspring):
                            new_offspring[k] += mutation
                            if new_offspring[k] < self.search_space[0]:
                                new_offspring[k] = self.search_space[0]
                            elif new_offspring[k] > self.search_space[1]:
                                new_offspring[k] = self.search_space[1]
                new_population[i] = new_offspring
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
        return population[0]