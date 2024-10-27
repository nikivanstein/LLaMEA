import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.population_size = 100
        self.population_mutation_rate = 0.01
        self.population_crossover_rate = 0.5

    def __call__(self, func):
        # Evaluate the function for each individual in the population
        fitnesses = [func(individual) for individual in self.evaluate_population()]

        # Select the best individuals
        best_individuals = sorted(zip(fitnesses, range(len(fitnesses))), key=lambda x: x[0], reverse=True)[:self.population_size]

        # Create a new population by crossover and mutation
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(best_individuals, 2)
            child = (parent1[0] + parent2[0]) / 2, (parent1[1] + parent2[1]) / 2
            if random.random() < self.population_mutation_rate:
                child[0] += random.uniform(-1, 1)
                child[1] += random.uniform(-1, 1)
            if random.random() < self.population_crossover_rate:
                child = (parent1[0] * parent2[0] + child[0] * parent2[0], parent1[1] * parent2[1] + child[1] * parent2[1])
            new_population.append(child)
        new_population = self.evaluate_population(new_population)

        # Replace the old population with the new one
        self.population = new_population

        # Check if the budget is reached
        if len(self.population) < self.budget:
            # If not, return the best individual found so far
            return self.search_space[0], self.search_space[1]
        else:
            # If the budget is reached, return the best individual found so far
            return self.search_space[0], self.search_space[1]

    def evaluate_population(self, population):
        fitnesses = [func(individual) for individual in population]
        return fitnesses

    def evaluate_fitness(self, individual, logger):
        # Evaluate the function at the individual
        func_value = func(individual)
        # Increment the function evaluations
        self.func_evaluations += 1
        # Check if the individual is within the budget
        if self.func_evaluations < self.budget:
            # If not, return the individual
            return individual
        # If the budget is reached, return the best individual found so far
        return self.search_space[0], self.search_space[1]

def func(individual):
    # This is a black box function that returns a random number between 0 and 1
    return random.random()

# Initialize the optimizer
optimizer = BlackBoxOptimizer(budget=100, dim=10)

# Run the optimizer
best_individual, best_fitness = optimizer(individual=[1, 2, 3, 4, 5])
print(f"Best individual: {best_individual}, Best fitness: {best_fitness}")