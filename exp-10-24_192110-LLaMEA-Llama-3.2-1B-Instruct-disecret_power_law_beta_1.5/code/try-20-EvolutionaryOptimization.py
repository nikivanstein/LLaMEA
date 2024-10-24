import numpy as np
import random
from scipy.optimize import minimize

class EvolutionaryOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = self.generate_population()

    def generate_population(self):
        # Initialize the population with random solutions
        population = []
        for _ in range(100):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(solution)
        return population

    def __call__(self, func):
        # Optimize the function using the given budget
        best_solution = None
        best_score = -np.inf
        for _ in range(self.budget):
            # Select the best individual from the population
            best_individual = self.select_best_individual(population)
            # Optimize the function using the selected individual
            score = func(best_individual)
            # Update the best solution if the score is better
            if score > best_score:
                best_solution = best_individual
                best_score = score
        return best_solution

    def select_best_individual(self, population):
        # Select the best individual based on the probability of refinement
        best_individual = population[0]
        probabilities = []
        for individual in population:
            probabilities.append(self.probability_of_refinement(individual, best_individual))
        probabilities = np.array(probabilities) / np.sum(probabilities)
        r = np.random.choice(len(population), p=probabilities)
        return population[r]

    def probability_of_refinement(self, individual, best_individual):
        # Refine the strategy based on the probability of improvement
        if individual == best_individual:
            return 0.05
        elif np.random.rand() < 0.5:
            return 0.1
        else:
            return 0.05

    def fitness_function(self, individual):
        # Evaluate the function using the given individual
        func = lambda x: x[0]**2 + x[1]**2
        return func(individual)

# Example usage:
budget = 100
dim = 2
optimization = EvolutionaryOptimization(budget, dim)
best_solution = optimization(func, np.random.uniform(-5.0, 5.0, dim))

# Print the results
print(f"Best solution: {best_solution}")
print(f"Best score: {optimization.fitness_function(best_solution)}")