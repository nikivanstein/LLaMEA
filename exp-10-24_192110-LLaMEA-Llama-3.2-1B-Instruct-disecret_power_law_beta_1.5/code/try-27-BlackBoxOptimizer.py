import numpy as np
import random

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.generate_initial_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.best_individual = None
        self.best_fitness = np.inf

    def generate_initial_population(self):
        population = []
        for _ in range(self.population_size):
            dim = random.randint(-5.0, 5.0)
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)
        return population

    def __call__(self, func):
        def evaluate_func(individual):
            return func(individual)

        for _ in range(self.budget):
            fitness = evaluate_func(self.population[0])
            if fitness < self.best_fitness:
                self.best_individual = self.population[0]
                self.best_fitness = fitness
            for i in range(1, self.population_size):
                fitness = evaluate_func(self.population[i])
                if fitness < self.best_fitness:
                    self.best_individual = self.population[i]
                    self.best_fitness = fitness
            self.population[0] = self.population[i]
            self.fitness_scores[i] = evaluate_func(self.population[0])
            if self.fitness_scores[i] < self.best_fitness:
                self.population[i] = self.population[0]
                self.fitness_scores[i] = self.best_fitness

        return self.fitness_scores

    def select_next_individual(self):
        if len(self.population) == 0:
            return self.population[0]
        if len(self.population) == 1:
            return self.population[0]
        return random.choices(self.population, weights=self.fitness_scores, k=1)[0]

    def mutate(self, individual):
        if random.random() < 0.05:
            dim = random.randint(-5.0, 5.0)
            individual = individual.copy()
            individual[dim] = random.uniform(-5.0, 5.0)
        return individual

# One-line description with main idea
# Novel metaheuristic algorithm for black box optimization using evolutionary strategy with mutation