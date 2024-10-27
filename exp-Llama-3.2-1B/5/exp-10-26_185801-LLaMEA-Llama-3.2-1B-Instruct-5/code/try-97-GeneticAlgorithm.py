import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.population = self.initialize_population()
        self.fitnesses = self.calculate_fitnesses(self.population)

    def initialize_population(self):
        return [self.generate_individual() for _ in range(self.budget)]

    def generate_individual(self):
        return tuple(random.uniform(-5.0, 5.0) for _ in range(self.dim))

    def calculate_fitnesses(self, population):
        fitnesses = []
        for individual in population:
            func_value = self.func_evaluations(individual)
            fitnesses.append(func_value)
        return fitnesses

    def __call__(self, func):
        while self.fitnesses[-1] < self.budget:
            for individual in self.population:
                func_value = self.func_evaluations(individual)
                if np.isnan(func_value) or np.isinf(func_value):
                    raise ValueError("Invalid function value")
                if func_value < 0 or func_value > 1:
                    raise ValueError("Function value must be between 0 and 1")
            self.population = self.select_population()
            self.fitnesses = self.calculate_fitnesses(self.population)
        return self.population[0]

    def select_population(self):
        # Selection strategy: Roulette wheel selection
        total_fitnesses = sum(self.fitnesses)
        selected_individuals = []
        for _ in range(self.budget):
            roulette_wheel = np.random.rand()
            cumulative_fitness = 0
            for fitness in self.fitnesses:
                cumulative_fitness += fitness
                if roulette_wheel <= cumulative_fitness:
                    selected_individuals.append(self.population.index(individual) for individual in self.population)
                    break
        return [self.population[i] for i in selected_individuals]

    def mutate(self, individual):
        # Mutation strategy: One-point mutation
        if random.random() < 0.05:
            mutated_individual = individual[:random.randint(0, self.dim)] + tuple(random.uniform(-5.0, 5.0) for _ in range(self.dim))
            return mutated_individual
        return individual

    def print_solution(self, solution):
        print("Solution:", solution)
        func_value = self.func_evaluations(solution)
        print("Function value:", func_value)

# Description: Genetic Algorithm for Black Box Optimization
# Code: 