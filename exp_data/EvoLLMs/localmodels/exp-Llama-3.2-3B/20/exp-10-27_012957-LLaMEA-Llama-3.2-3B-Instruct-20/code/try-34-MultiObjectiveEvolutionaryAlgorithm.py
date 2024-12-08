import numpy as np
import random
from scipy.optimize import minimize

class MultiObjectiveEvolutionaryAlgorithm:
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

    def probabilistic_line_search(self, func, x, step_size):
        if random.random() < 0.2:
            new_x = x + step_size
            new_fitness = func(new_x)
            if new_fitness < func(x):
                x = new_x
        return x

    def optimize(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
            selected_individuals = self.selection()
            offspring = []
            for _ in range(int(self.population_size/2)):
                parent1, parent2 = random.sample(selected_individuals, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                child = self.probabilistic_line_search(func, child, 0.1)
                offspring.append(child)
            self.population = np.concatenate((selected_individuals, offspring))
        return np.mean(self.fitness_values)

# Example usage:
def func(x):
    return np.sum(x**2)

moecbm = MultiObjectiveEvolutionaryAlgorithm(budget=100, dim=10)
best_fitness = moecbm.optimize(func)
print("Best fitness:", best_fitness)