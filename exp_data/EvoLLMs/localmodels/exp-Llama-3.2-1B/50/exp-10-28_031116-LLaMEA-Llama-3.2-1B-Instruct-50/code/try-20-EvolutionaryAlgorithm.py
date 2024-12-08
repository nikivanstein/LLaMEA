import numpy as np
import random

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = []
        self.fitness_scores = []

    def __call__(self, func):
        # Generate initial population
        population = [func(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(100)]

        # Evolve population for the specified number of generations
        for _ in range(1000):
            # Select parents based on fitness scores
            parents = random.choices(self.population, weights=self.fitness_scores, k=100)

            # Crossover (reproduce) offspring
            offspring = []
            for _ in range(100):
                parent1, parent2 = random.sample(parents, 2)
                child = (parent1 + parent2) / 2
                offspring.append(child)

            # Mutate offspring
            for i in range(len(offspring)):
                if random.random() < 0.1:
                    offspring[i] += random.uniform(-1, 1)

            # Replace least fit individuals with offspring
            self.population = offspring

            # Update fitness scores
            self.fitness_scores = [func(x) for x in self.population]

        # Select best individual
        best_individual = max(self.population, key=func)

        return best_individual

    def fitness(self, func, individual):
        # Evaluate function at given individual
        return func(individual)

# Define a black box function
def func(x):
    return np.sin(x)

# Create an instance of the evolutionary algorithm
ea = EvolutionaryAlgorithm(budget=100, dim=10)

# Evaluate the black box function
best_individual = ea(__call__(func))
print("Best individual:", best_individual)
print("Best fitness score:", ea.fitness(func, best_individual))

# Update the population with the new best individual
best_individual = ea(__call__(func))
ea.population = [best_individual]

# Evaluate the black box function again
best_individual = ea(__call__(func))
print("New best individual:", best_individual)
print("New best fitness score:", ea.fitness(func, best_individual))

# Update the population with the new best individual again
best_individual = ea(__call__(func))
ea.population = [best_individual]

# Evaluate the black box function again
best_individual = ea(__call__(func))
print("Final best individual:", best_individual)
print("Final best fitness score:", ea.fitness(func, best_individual))