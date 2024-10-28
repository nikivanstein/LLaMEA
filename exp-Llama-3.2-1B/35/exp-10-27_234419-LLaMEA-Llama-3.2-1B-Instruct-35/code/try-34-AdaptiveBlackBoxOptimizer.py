import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func = None
        self.search_space = None
        self.sample_size = None
        self.sample_indices = None
        self.local_search = False
        self.population_size = 100

    def __call__(self, func):
        if self.func is None:
            self.func = func
            self.search_space = np.random.uniform(-5.0, 5.0, self.dim)
            self.sample_size = 1
            self.sample_indices = None

        if self.budget <= 0:
            raise ValueError("Budget is less than or equal to zero")

        # Initialize the population with random individuals
        self.population = self.initialize_population()

        while True:
            # Evaluate the fitness of each individual
            fitness = [self.evaluate_fitness(individual) for individual in self.population]

            # Select the best individual based on the probability of 0.35
            selected_individuals = np.random.choice(self.population_size, size=self.population_size, replace=False, p=fitness)

            # Perform local search on the selected individuals
            selected_individuals = self.perform_local_search(selected_individuals)

            # Replace the old population with the new one
            self.population = selected_individuals

            # Check if the budget is reached
            if len(self.population) >= self.budget:
                break

        return func(self.population[0])

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.random.uniform(self.search_space, size=self.dim)
            population.append(individual)
        return population

    def evaluate_fitness(self, individual):
        func = self.func
        return func(individual)

    def perform_local_search(self, selected_individuals):
        # For now, we'll just perform a simple random search
        # In the future, we can use more advanced local search algorithms
        best_individual = None
        best_fitness = float('inf')
        for individual in selected_individuals:
            fitness = self.evaluate_fitness(individual)
            if fitness < best_fitness:
                best_individual = individual
                best_fitness = fitness
        return selected_individuals[:best_individual]

# One-line description with the main idea
# Adaptive Black Box Optimization using Genetic Algorithm
# to solve black box optimization problems with a wide range of tasks
# and a population of individuals with varying search spaces and fitness functions