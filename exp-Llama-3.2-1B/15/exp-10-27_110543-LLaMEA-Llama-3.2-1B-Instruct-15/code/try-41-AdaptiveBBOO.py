import numpy as np
import random

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []
        self.population_strategy = None

    def __call__(self, func):
        def eval_func(x):
            return func(x)

        def evaluate_budget(func, x, budget):
            if budget <= 0:
                raise ValueError("Budget cannot be zero or negative")
            return np.sum([eval_func(x + np.random.normal(0, 1, size=self.dim)) for _ in range(budget)])

        def mutate(individual, mutation_rate):
            if mutation_rate < 0.1:
                mutated_individual = individual.copy()
                for i in range(self.dim):
                    if random.random() < mutation_rate:
                        mutated_individual[i] += np.random.normal(0, 1)
                return mutated_individual
            else:
                return individual

        def crossover(parent1, parent2):
            if random.random() < 0.5:
                child = parent1[:self.dim//2] + parent2[self.dim//2:]
            else:
                child = parent1 + parent2[self.dim//2:]
            return child

        def next_generation(population):
            new_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                parent1 = population[i]
                parent2 = population[np.random.randint(0, self.population_size)]
                child = crossover(parent1, parent2)
                child = mutate(child, 0.1)
                new_population[i] = child
            return new_population

        def optimize(func, budget):
            for _ in range(budget):
                for i in range(self.population_size):
                    fitness = evaluate_budget(func, self.population[i], budget)
                    self.fitnesses[i] = fitness
                    self.population_history.append(self.population[i])

            # Select the fittest individuals
            self.population = self.population[np.argsort(self.fitnesses, axis=0)]
            self.fitnesses = self.fitnesses[np.argsort(self.fitnesses, axis=0)]

            # Evolve the population
            for _ in range(100):
                next_population = next_generation(self.population)
                self.population = next_population

            # Update the population strategy
            if self.population_strategy is None:
                self.population_strategy = self.budget / self.population_size
            else:
                self.population_strategy *= (1 + self.population_strategy * 0.1)

        return optimize

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies