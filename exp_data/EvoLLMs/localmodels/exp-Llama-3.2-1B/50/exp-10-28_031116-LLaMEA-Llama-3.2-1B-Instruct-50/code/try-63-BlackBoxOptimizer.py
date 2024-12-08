import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.population_history = []

    def __call__(self, func):
        # Generate a random initial population
        population = self.generate_initial_population(self.dim)

        # Evaluate the initial population
        scores = [func(x) for x in population]

        # Select the best individual
        best_individual = self.select_best_individual(scores)

        # Refine the population
        self.refine_population(population, best_individual)

        # Return the best individual
        return best_individual

    def generate_initial_population(self, dim):
        # Generate a random population with a search space between -5.0 and 5.0
        return [np.random.uniform(-5.0, 5.0, dim) for _ in range(100)]

    def select_best_individual(self, scores):
        # Select the best individual based on the budget constraint
        return self.select_individual(scores, self.budget)

    def refine_population(self, population, best_individual):
        # Refine the population using a novel metaheuristic algorithm
        for _ in range(10):
            # Generate a new individual using the best individual and a random perturbation
            new_individual = best_individual + random.uniform(-1.0, 1.0)

            # Evaluate the new individual
            score = func(new_individual)

            # If the new individual is better, add it to the population
            if score > func(best_individual):
                population.append(new_individual)
                self.population_history.append([best_individual, score])

                # Update the best individual
                best_individual = new_individual

    def select_individual(self, scores, budget):
        # Select an individual based on the probability of success
        probabilities = [score / budget for score in scores]
        return random.choices(population, weights=probabilities, k=1)[0]

def func(x):
    # Black box function
    return np.sin(x)

optimizer = BlackBoxOptimizer(100, 10)
best_individual = optimizer(func)
print("Best individual:", best_individual)
print("Best score:", func(best_individual))