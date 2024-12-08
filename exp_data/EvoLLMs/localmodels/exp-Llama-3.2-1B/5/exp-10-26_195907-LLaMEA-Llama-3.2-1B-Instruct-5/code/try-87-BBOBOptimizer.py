# Description: Novel Metaheuristic Algorithm for Black Box Optimization (NMBAO)
# Code: 
# ```python
import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func, budget):
        if budget <= 0:
            raise ValueError("Budget must be greater than zero")
        
        # Initialize population with random individuals
        population = self.generate_population(self.budget)

        # Evaluate fitness for each individual and select the best ones
        for individual in population:
            fitness = self.evaluate_fitness(individual, func)
            population[fitness < self.budget] = individual

        # Refine the population based on the selected solution
        selected_individual = self.select_solution(population, func, budget)

        # Create a new population with the selected individual
        new_population = self.generate_population(self.budget)

        # Replace the old population with the new one
        population = new_population

        return selected_individual, population

    def generate_population(self, budget):
        population = []
        for _ in range(budget):
            individual = np.random.choice(self.search_space.shape[0], self.dim, replace=False)
            population.append(individual)
        return population

    def evaluate_fitness(self, individual, func):
        fitness = func(individual)
        return fitness

    def select_solution(self, population, func, budget):
        # Refine the population based on the selected solution
        selected_individual = population[np.random.randint(0, len(population), size=self.dim)]
        while np.linalg.norm(func(selected_individual)) < budget / 2:
            selected_individual = population[np.random.randint(0, len(population), size=self.dim)]
        return selected_individual

# Example usage:
budget = 100
dim = 10
optimizer = BBOBOptimizer(budget, dim)
selected_individual, population = optimizer(__call__, budget)
print("Selected Individual:", selected_individual)
print("Population:", population)