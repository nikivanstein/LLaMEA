# Description: Novel metaheuristic algorithm to optimize black box functions using a population-based approach
# Code: 
# ```python
import numpy as np
from scipy.optimize import minimize
import random

class BBOBBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, 100)
        self.func_evaluations = 0
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        return [random.uniform(self.search_space) for _ in range(self.population_size)]

    def __call__(self, func):
        def wrapper(x):
            if self.func_evaluations >= self.budget:
                raise Exception("Optimization limit reached")
            return func(x)

        self.func_evaluations += 1
        try:
            result = minimize(wrapper, self.search_space[0], method="SLSQP", bounds=[(x, x) for x in self.search_space])
            return result.x
        except Exception as e:
            print(f"Error: {e}")
            return None

    def select_solution(self, population):
        selected_individuals = random.sample(population, self.population_size)
        selected_individuals = [individual for individual in selected_individuals if self.evaluate_fitness(individual)]
        if not selected_individuals:
            return None
        return selected_individuals

    def evaluate_fitness(self, individual):
        return individual

    def mutate(self, individual):
        if random.random() < 0.2:
            individual = individual + random.uniform(-1, 1)
        return individual

    def __str__(self):
        return f"Population size: {self.population_size}\nPopulation: {self.population}"

# Example usage:
optimizer = BBOBBlackBoxOptimizer(1000, 10)
func = lambda x: x**2
result = optimizer(func)
print(result)

# Select the solution with a probability of 0.2
selected_individual = optimizer.select_solution(optimizer.population)
print(f"Selected individual: {selected_individual}")

# Evaluate the fitness of the selected individual
fitness = optimizer.evaluate_fitness(selected_individual)
print(f"Fitness: {fitness}")

# Mutate the selected individual with a probability of 20%
mutated_individual = selected_individual.copy()
mutated_individual = optimizer.mutate(mutated_individual)
print(f"Mutated individual: {mutated_individual}")

# Update the population with the mutated individual
optimizer.population = [individual for individual in optimizer.population if individual == mutated_individual]