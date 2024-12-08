import numpy as np
import random

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        return [self.generate_individual() for _ in range(self.population_size)]

    def generate_individual(self):
        return np.random.uniform(self.search_space)

    def mutate(self, individual):
        idx = random.randint(0, self.dim - 1)
        individual[idx] = random.uniform(-5.0, 5.0)
        return individual

    def crossover(self, parent1, parent2):
        idx1 = random.randint(0, self.dim - 1)
        idx2 = random.randint(0, self.dim - 1)
        child = np.concatenate((parent1[:idx1], parent2[idx2:]))
        return child

    def evaluate_fitness(self, individual):
        func_value = self.func_evaluations(individual)
        if np.isnan(func_value) or np.isinf(func_value):
            raise ValueError("Invalid function value")
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        return func_value

    def selection(self):
        fitnesses = [self.evaluate_fitness(individual) for individual in self.population]
        sorted_indices = np.argsort(fitnesses)
        return [self.population[i] for i in sorted_indices]

    def mutation_rate(self):
        return 0.01

    def crossover_rate(self):
        return 0.5

    def mutate_rate(self):
        return 0.1

    def __call__(self, func):
        population = self.population
        while len(population) > 0:
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = random.sample(population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            population = new_population
            population = self.selection()
            population = [self.mutate_rate() * individual + self.crossover_rate() * population[i] for i, individual in enumerate(population)]
        return self.evaluate_fitness(population[0])

# Initialize the HEBBO algorithm
hebbbo = HEBBO(100, 5)

# Run the algorithm and print the result
print(hebbbo())