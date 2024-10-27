import numpy as np
import random

class MetaHEBBO:
    def __init__(self, budget, dim, mutation_rate, alpha, beta):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.beta = beta
        self.population = self.initialize_population()

    def initialize_population(self):
        # Initialize the population with random solutions
        population = []
        for _ in range(1000):
            individual = np.random.uniform(self.search_space)
            population.append(individual)
        return population

    def __call__(self, func):
        # Evaluate the function for each individual in the population
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            new_individual = self.evaluate_fitness(func, self.population)
            if np.isnan(new_individual) or np.isinf(new_individual):
                raise ValueError("Invalid function value")
            if new_individual < 0 or new_individual > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

    def evaluate_fitness(self, func, population):
        # Evaluate the fitness of each individual in the population
        fitness = np.zeros(len(population))
        for i, individual in enumerate(population):
            fitness[i] = func(individual)
        return fitness

    def mutate(self, individual):
        # Randomly mutate an individual in the population
        mutated_individual = individual.copy()
        if random.random() < self.mutation_rate:
            mutated_individual[random.randint(0, self.dim - 1)] += np.random.uniform(-1, 1)
        return mutated_individual

    def select_solution(self, func, population, fitness):
        # Select the solution with the highest fitness
        selected_individual = np.argmax(fitness)
        selected_individual = population[selected_individual]
        return selected_individual

    def update_solution(self, func, population, fitness):
        # Update the selected solution using the metaheuristic algorithm
        new_individual = self.select_solution(func, population, fitness)
        new_individual = self.evaluate_fitness(func, new_individual)
        if np.isnan(new_individual) or np.isinf(new_individual):
            raise ValueError("Invalid function value")
        if new_individual < 0 or new_individual > 1:
            raise ValueError("Function value must be between 0 and 1")
        self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return new_individual

    def run(self, func, population_size, mutation_rate, alpha, beta):
        # Run the metaheuristic algorithm
        for _ in range(100):
            new_individual = self.update_solution(func, population, self.evaluate_fitness(func, population))
            if np.isnan(new_individual) or np.isinf(new_individual):
                raise ValueError("Invalid function value")
            if new_individual < 0 or new_individual > 1:
                raise ValueError("Function value must be between 0 and 1")
        return new_individual

# Description: A metaheuristic hybrid of Heuristic and Evolutionary Algorithms
# Code: 