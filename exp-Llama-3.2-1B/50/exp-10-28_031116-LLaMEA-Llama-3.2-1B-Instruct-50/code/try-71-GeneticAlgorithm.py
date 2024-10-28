import numpy as np
import random
import operator

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.generate_initial_population()
        self.fitness_scores = []
        self.best_solution = None

    def generate_initial_population(self):
        population = []
        for _ in range(self.population_size):
            dim = random.randint(-5, 5)
            solution = np.random.uniform(-5, 5, self.dim)
            population.append(solution)
        return population

    def __call__(self, func):
        def evaluate(func, solution):
            return func(solution)

        def fitness(solution, func):
            return evaluate(func, solution)

        def selection(population, func):
            fitness_scores = [fitness(solution, func) for solution in population]
            sorted_indices = np.argsort(fitness_scores)
            return population[sorted_indices[:self.population_size//2]]

        def crossover(parent1, parent2):
            child = parent1.copy()
            for _ in range(self.dim):
                if random.random() < 0.5:
                    child[_] = random.uniform(-5, 5)
                else:
                    child[_] = parent2[_]
            return child

        def mutation(solution):
            if random.random() < 0.1:
                index = random.randint(0, self.dim-1)
                solution[index] = random.uniform(-5, 5)
            return solution

        population = selection(population, evaluate)
        for _ in range(self.budget):
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            child = mutation(child)
            population.append(child)

        self.population = population
        self.fitness_scores = [fitness(solution, evaluate) for solution in self.population]

        self.best_solution = min(self.population, key=fitness)

    def __str__(self):
        return f"Genetic Algorithm with Adaptive Crossover and Mutation\nBest Solution: {self.best_solution}\nFitness: {self.fitness_scores[self.best_solution]}"

# Description: Evolutionary Optimization using Genetic Algorithm with Adaptive Crossover and Mutation
# Code: 