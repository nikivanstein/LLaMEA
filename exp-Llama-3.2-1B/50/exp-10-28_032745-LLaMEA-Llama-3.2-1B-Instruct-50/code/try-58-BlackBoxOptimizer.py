import numpy as np
from scipy.optimize import minimize
import random
import copy

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func = lambda x: x[0] * x[1]  # Example black box function

    def __call__(self, func, initial_guess, iterations):
        population_size = 100
        for _ in range(iterations):
            population = [copy.deepcopy(initial_guess) for _ in range(population_size)]
            fitnesses = []
            for individual in population:
                fitness = self.evaluate_fitness(individual)
                fitnesses.append(fitness)
            fitnesses.sort(reverse=True)
            fitnesses = fitnesses[:self.budget]
            for individual in population:
                fitness = self.evaluate_fitness(individual)
                if fitness in fitnesses:
                    population.remove(individual)
                    population.append(individual)
            if len(population) > 0:
                new_population = []
                while len(new_population) < population_size:
                    parent1, parent2 = random.sample(population, 2)
                    child = self.crossover(parent1, parent2)
                    child = self.mutation(child)
                    new_population.append(child)
                population = new_population
            if len(population) > 0:
                best_individual, best_fitness = self.select_best(population, fitnesses)
                return best_individual, best_fitness

    def evaluate_fitness(self, individual):
        return self.func(individual)

    def crossover(self, parent1, parent2):
        child = parent1[:self.dim]
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent2[i]
        return child

    def mutation(self, individual):
        for i in range(self.dim):
            if random.random() < 0.1:
                individual[i] += random.uniform(-0.1, 0.1)
        return individual

    def select_best(self, population, fitnesses):
        best_individual = None
        best_fitness = float('-inf')
        for individual in population:
            fitness = self.evaluate_fitness(individual)
            if fitness > best_fitness:
                best_individual = individual
                best_fitness = fitness
        return best_individual, best_fitness

# One-line description with the main idea
# Novel metaheuristic algorithm for black box optimization using Enhanced Genetic Algorithm