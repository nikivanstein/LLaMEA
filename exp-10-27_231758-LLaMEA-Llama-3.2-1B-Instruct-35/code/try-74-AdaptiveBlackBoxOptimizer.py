# AdaptiveBlackBoxOptimizer: A novel metaheuristic algorithm for solving black box optimization problems
# Description: This algorithm optimizes a given function using a population-based approach, with a refined strategy to adapt to the evolution process
# Code: 
# ```python
import numpy as np
import random

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim, mutation_prob=0.01, crossover_prob=0.5):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.func_values = None
        self.population_size = 100
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness_scores = np.zeros(self.population_size)

    def __call__(self, func):
        if self.func_values is None:
            self.func_evals = self.budget
            self.func_values = np.zeros(self.dim)
            for _ in range(self.func_evals):
                func(self.func_values)
        else:
            while self.func_evals > 0:
                idx = np.argmin(np.abs(self.func_values))
                self.func_values[idx] = func(self.func_values[idx])
                self.func_evals -= 1
                if self.func_evals == 0:
                    break

        self.fitness_scores = np.array([np.sqrt(np.mean((func(self.func_values) - func(np.array([0]))**2) / (func(np.array([0]))**2))**2) for func in self.population])

        # Select the fittest individuals for the next generation
        self.population = self.population[np.argsort(self.fitness_scores)]

        # Perform mutation and crossover
        self.population = self.population[random.sample(self.population, self.population_size//2)]
        self.population = self.population + 2 * random.uniform(-1, 1) * self.population[random.sample(self.population, self.population_size//2)]
        self.population = self.population + 2 * random.uniform(-1, 1) * self.population[random.sample(self.population, self.population_size//2)]

        # Refine the strategy based on the probability of mutation and crossover
        if random.random() < self.crossover_prob:
            self.population = self.population[random.sample(self.population, 2)]
        if random.random() < self.mutation_prob:
            self.population = self.population[random.sample(self.population, 2)] + 2 * random.uniform(-1, 1) * self.population[random.sample(self.population, 2)]

        # Update the population and fitness scores
        self.population = np.array(self.population)
        self.fitness_scores = np.array([np.sqrt(np.mean((func(self.func_values) - func(np.array([0]))**2) / (func(np.array([0]))**2))**2) for func in self.population])

        # Update the best individual
        self.population[self.fitness_scores.argmax()] = self.func_values

        return self.population, self.fitness_scores

# Test the algorithm
optimizer = AdaptiveBlackBoxOptimizer(budget=1000, dim=10)
optimizer(__call__)

# Print the results
print("Optimal function:", optimizer.__call__(np.array([-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0])))
print("Best fitness score:", optimizer.__call__(np.array([-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0])))

# Get the best individual
best_individual = optimizer.population[np.argmin(np.abs(optimizer.func_values))]

# Get the best fitness score
best_fitness_score = np.sqrt(np.mean((best_individual - np.array([0]))**2 / (np.array([0]))**2))

print("Best individual:", best_individual)
print("Best fitness score:", best_fitness_score)