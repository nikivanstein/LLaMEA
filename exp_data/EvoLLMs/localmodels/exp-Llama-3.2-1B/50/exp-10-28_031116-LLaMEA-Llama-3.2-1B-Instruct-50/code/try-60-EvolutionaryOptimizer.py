import numpy as np
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class EvolutionaryOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.generate_initial_population()
        self.fitness_scores = [0] * self.population_size

    def generate_initial_population(self):
        return [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

    def __call__(self, func):
        def fitness_func(x):
            return func(x)

        # Evaluate the function a specified number of times within the budget
        num_evaluations = min(self.budget, len(self.fitness_scores))
        evaluations = np.random.choice([0, 1], size=num_evaluations, p=[0.9, 0.1])

        # Initialize the population with the evaluations
        self.population = [x for i, x in enumerate(self.population) if evaluations[i] == 0]
        self.fitness_scores = [fitness_func(x) for x in self.population]

        # Select the fittest individuals to reproduce
        self.population = self.select_fittest(population_size=self.population_size)

        # Perform crossover and mutation to create new offspring
        offspring = []
        for _ in range(self.population_size - self.population_size // 2):
            parent1, parent2 = random.sample(self.population, 2)
            child = (parent1 + parent2) / 2
            if random.random() < 0.5:
                child = child + random.uniform(-5.0, 5.0)
            offspring.append(child)

        # Replace the least fit individuals with the new offspring
        self.population = self.population[:self.population_size // 2] + offspring

        # Evaluate the new population
        new_evaluations = min(self.budget, len(self.fitness_scores))
        new_population = [x for i, x in enumerate(self.population) if evaluations[i] == 1]
        self.fitness_scores = [fitness_func(x) for x in new_population]

        # Update the population with the new evaluations
        self.population = self.population[:self.population_size]
        self.fitness_scores = [fitness_func(x) for x in self.population]

    def select_fittest(self, population_size=50):
        scores = [self.fitness_scores[i] for i in range(len(self.fitness_scores))]
        idx = np.argsort(scores)[-population_size:]
        return [self.population[i] for i in idx]

    def __str__(self):
        return f"Fitness scores: {self.fitness_scores}"