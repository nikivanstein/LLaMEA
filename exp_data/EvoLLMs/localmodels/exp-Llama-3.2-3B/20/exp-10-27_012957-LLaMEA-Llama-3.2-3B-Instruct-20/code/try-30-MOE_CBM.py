import numpy as np
import random

class MOE_CBM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.fitness_values = []
        self.population = self.initialize_population()

    def initialize_population(self):
        population = np.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            for j in range(self.dim):
                population[i, j] = random.uniform(-5.0, 5.0)
        return population

    def evaluate(self, func):
        self.fitness_values = []
        for individual in self.population:
            fitness = func(individual)
            self.fitness_values.append(fitness)

    def selection(self):
        sorted_indices = np.argsort(self.fitness_values)
        selected_indices = sorted_indices[:int(self.population_size/2)]
        selected_individuals = self.population[sorted_indices[:int(self.population_size/2)]]
        return selected_individuals

    def crossover(self, parent1, parent2):
        child = parent1 + parent2 * (parent2 - parent1)
        return child

    def mutation(self, individual):
        for i in range(self.dim):
            if random.random() < self.mutation_rate:
                individual[i] += random.uniform(-1.0, 1.0)
                individual[i] = max(-5.0, min(5.0, individual[i]))
        return individual

    def refine_solution(self, selected_individuals):
        refined_population = selected_individuals.copy()
        for i in range(self.population_size):
            if random.random() < 0.2:
                parent1, parent2 = random.sample(selected_individuals, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                refined_population[i] = child
        return refined_population

    def optimize(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
            selected_individuals = self.selection()
            refined_population = self.refine_solution(selected_individuals)
            offspring = refined_population
            self.population = np.concatenate((selected_individuals, offspring))
        return np.mean(self.fitness_values)

# Example usage:
def func(x):
    return np.sum(x**2)

moecbm = MOE_CBM(budget=100, dim=10)
best_fitness = moecbm.optimize(func)
print("Best fitness:", best_fitness)