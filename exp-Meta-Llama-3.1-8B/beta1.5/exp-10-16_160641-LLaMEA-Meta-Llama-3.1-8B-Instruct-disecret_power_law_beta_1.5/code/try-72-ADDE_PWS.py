import numpy as np
import random

class ADDE_PWS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.population_size = 50
        self.num_parents = int(0.2 * self.population_size)
        self.num_offspring = self.population_size - self.num_parents
        self.diffusion_rate = 0.1
        self.driven_evolution_rate = 0.05
        self.probability_threshold = 0.5

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

    def probability_weighted_selection(self, population, func):
        fitness = self.evaluate(func, population)
        sorted_idx = np.argsort(fitness)
        probabilities = np.linspace(0, 1, self.population_size)
        probabilities[sorted_idx] = np.sort(probabilities[sorted_idx])
        new_population = population.copy()
        for i in range(self.num_parents):
            parent_idx = np.random.choice(self.population_size, p=probabilities)
            new_population[i] = population[parent_idx]
        return new_population

    def __call__(self, func):
        population = self.initialize_population()
        for _ in range(self.budget):
            population = self.probability_weighted_selection(population, func)
            population = self.diffusion(population)
            population = self.driven_evolution(population, func)
        return population[0]