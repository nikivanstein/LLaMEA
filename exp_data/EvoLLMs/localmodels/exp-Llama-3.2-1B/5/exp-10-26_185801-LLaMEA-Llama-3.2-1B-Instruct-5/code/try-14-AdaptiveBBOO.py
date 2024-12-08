import numpy as np
import random

class AdaptiveBBOO:
    def __init__(self, budget, dim, mutation_rate, learning_rate):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.mutation_rate = mutation_rate
        self.learning_rate = learning_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(1000):
            individual = np.random.uniform(self.search_space)
            population.append(individual)
        return population

    def __call__(self, func, mutation_rate):
        while self.func_evaluations < self.budget:
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")

            # Select an individual based on fitness
            fitness = [self.evaluate_fitness(individual) for individual in self.population]
            selected_index = np.random.choice(len(self.population), p=fitness)

            # Perform mutation
            mutated_individual = self.population[selected_index][0]
            if random.random() < mutation_rate:
                mutated_individual = np.random.uniform(self.search_space)

            # Evaluate new individual
            new_individual = np.random.uniform(self.search_space)
            new_fitness = self.evaluate_fitness(new_individual)

            # Update population
            self.population[selected_index] = [new_individual, new_fitness]
            self.func_evaluations += 1

            # Update the best individual
            if new_fitness > self.search_space[0]:
                self.population[0] = [new_individual, new_fitness]

            # Update the population's fitness
            for i in range(1, len(self.population)):
                self.population[i] = [self.evaluate_fitness(individual), self.evaluate_fitness(self.population[i][1])]

        return self.search_space[0]

    def evaluate_fitness(self, individual):
        func_value = func(individual)
        if np.isnan(func_value) or np.isinf(func_value):
            raise ValueError("Invalid function value")
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        return func_value

# Example usage:
def func(x):
    return x**2

adaptive_bboo = AdaptiveBBOO(budget=100, dim=5, mutation_rate=0.01, learning_rate=0.1)
print(adaptive_bboo(adaptive_bboo, func, mutation_rate=0.01))