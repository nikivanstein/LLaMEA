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
        self.refine_strategy = False

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

    def refine_strategy(self):
        if self.refine_strategy:
            new_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                parent1, parent2 = random.sample(self.population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                if random.random() < 0.2:
                    child = self.refine_child(child)
                new_population[i] = child
            self.population = new_population
            self.refine_strategy = False

    def refine_child(self, child):
        # Refine the child by changing one of its dimensions
        index = random.randint(0, self.dim - 1)
        child[index] += random.uniform(-1.0, 1.0)
        child[index] = max(-5.0, min(5.0, child[index]))
        return child

    def optimize(self, func):
        for _ in range(self.budget):
            self.evaluate(func)
            selected_individuals = self.selection()
            offspring = []
            for _ in range(int(self.population_size/2)):
                parent1, parent2 = random.sample(selected_individuals, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutation(child)
                if random.random() < self.crossover_rate:
                    child = self.crossover(child, parent2)
                offspring.append(child)
            self.population = np.concatenate((selected_individuals, offspring))
            self.refine_strategy = True
        return np.mean(self.fitness_values)

# Example usage:
def func(x):
    return np.sum(x**2)

moecbm = MOE_CBM(budget=100, dim=10)
best_fitness = moecbm.optimize(func)
print("Best fitness:", best_fitness)