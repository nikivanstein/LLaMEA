import random
import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.generate_initial_population()
        self.fitness_values = np.zeros((self.population_size, self.dim))
        self.search_space = np.linspace(-5.0, 5.0, self.dim)

    def generate_initial_population(self):
        return np.random.uniform(self.search_space, self.search_space, (self.population_size, self.dim))

    def fitness_function(self, func, x):
        return func(x)

    def __call__(self, func):
        def objective(x):
            return self.fitness_function(func, x)

        def fitness(x):
            return objective(x)

        def selection(self, fitness_values):
            return np.random.choice(self.population_size, self.population_size, p=fitness_values)

        def crossover(self, parent1, parent2):
            child = np.zeros(self.dim)
            for i in range(self.dim):
                child[i] = (parent1[i] + parent2[i]) / 2
            return child

        def mutation(self, child):
            for i in range(self.dim):
                if random.random() < 0.1:
                    child[i] += random.uniform(-1, 1)
            return child

        def selection_with_crossover(self, fitness_values, parent1, parent2):
            selected_indices = selection(fitness_values)
            child1 = parent1[selected_indices]
            child2 = parent2[selected_indices]
            child = crossover(child1, child2)
            child = mutation(child)
            return child

        def selection_with_crossover_and_mutation(self, fitness_values, parent1, parent2):
            selected_indices = selection(fitness_values)
            child1 = parent1[selected_indices]
            child2 = parent2[selected_indices]
            child = crossover(child1, child2)
            child = mutation(child)
            return child

        def evolution(self, population, fitness_values, num_generations):
            for _ in range(num_generations):
                population = self.select(population, fitness_values)
                population = self.crossover(population)
                population = self.mutation(population)
            return population

        def select(self, fitness_values, num_indices):
            fitness_values = fitness_values / np.max(fitness_values)
            selected_indices = np.random.choice(self.population_size, num_indices, p=fitness_values)
            return self.population[selected_indices]

        def crossover(self, parent1, parent2):
            child = np.zeros(self.dim)
            for i in range(self.dim):
                child[i] = (parent1[i] + parent2[i]) / 2
            return child

        def mutation(self, child):
            for i in range(self.dim):
                if random.random() < 0.1:
                    child[i] += random.uniform(-1, 1)
            return child

        def mutate(self, population):
            for i in range(self.population_size):
                if random.random() < 0.1:
                    population[i] += random.uniform(-1, 1)
            return population

        def evolve(self, population, fitness_values, num_generations):
            population = self.select(population, fitness_values)
            for _ in range(num_generations):
                population = self.crossover(population)
                population = self.mutation(population)
            return population

        def run(self, func):
            while True:
                population = self.evolve(self.fitness_values, self.fitness_function, 100)
                if all(self.fitness_values >= 0.5):
                    break
            return population

        def select_best(self, population, fitness_values):
            return self.select(population, fitness_values)

        def select_worst(self, population, fitness_values):
            return self.select(population, fitness_values)

        def select_best_worst(self, population, fitness_values):
            return self.select_worst(population, fitness_values)

    def run_multiple(self, func, num_evaluations, num_iterations):
        results = []
        for _ in range(num_iterations):
            population = self.run(func)
            results.append((self.select_best(population, self.fitness_values), self.select_worst(population, self.fitness_values), self.select_best_worst(population, self.fitness_values)))
        return results

# Description: Black Box Optimization using Adaptive Evolutionary Strategies
# Code: 