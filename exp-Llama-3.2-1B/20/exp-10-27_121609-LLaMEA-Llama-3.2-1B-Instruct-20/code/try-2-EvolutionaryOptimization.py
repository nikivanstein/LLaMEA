import numpy as np
import random
import math
from scipy.optimize import minimize

class EvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []

    def __call__(self, func, bounds, mutation_prob):
        def objective(x):
            return func(x)

        def bounds_check(x):
            return (x.min() - bounds[0][0], x.max() + bounds[1][0])

        def mutate(x):
            if random.random() < mutation_prob:
                idx = random.randint(0, self.dim - 1)
                x[idx] = random.uniform(bounds[idx][0], bounds[idx][1])
            return x

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = mutate(x)

        # Select the best individual based on fitness and bounds
        selected_individual = np.argmax(self.fitnesses)
        selected_individual_bounds = bounds[selected_individual]

        # Refine the solution using probability 0.2
        refined_individual = selected_individual_bounds
        if random.random() < 0.2:
            refined_individual = mutate(refined_individual)

        # Update the population history
        self.population_history.append((selected_individual, refined_individual))

        return objective(refined_individual)

# Example usage
func = lambda x: x**2
bounds = (-5.0, 5.0)
evolutionary_optimization = EvolutionaryOptimization(100, 10)
best_solution = -np.inf
best_fitness = -np.inf

for _ in range(1000):
    solution = evolutionary_optimization(func, bounds, 0.1)
    best_solution = min(best_solution, solution)
    best_fitness = min(best_fitness, solution)

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)