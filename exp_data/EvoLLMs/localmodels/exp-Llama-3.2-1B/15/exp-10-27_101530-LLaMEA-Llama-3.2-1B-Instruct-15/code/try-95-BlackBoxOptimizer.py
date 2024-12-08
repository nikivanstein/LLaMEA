import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        def mutate(individual):
            if random.random() < 0.1:  # Refine the strategy with a 10% mutation rate
                new_individual = individual[:self.dim] + [random.uniform(-self.search_space[0], self.search_space[1]) for _ in range(self.dim)]
                return new_individual
            return individual

        def crossover(parent1, parent2):
            if random.random() < 0.5:  # Perform crossover with a 50% probability
                child = parent1[:self.dim] + parent2[self.dim:]
                return child
            return parent1

        def selection(fitness_values):
            return sorted(fitness_values)[::-1]

        def evolve_population(population, mutation_rate, crossover_rate, selection_rate):
            new_population = []
            for _ in range(population):
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                child = crossover(parent1, parent2)
                child = mutate(child)
                child = selection(fitness_values[child])
                new_population.append(child)
            return new_population

        self.population = evolve_population(population=self.population, mutation_rate=0.01, crossover_rate=0.5, selection_rate=0.1)

        # Evaluate the best individual in the new population
        best_individual = max(self.population, key=lambda individual: individual[-1])
        best_fitness = individual[-1]
        # Update the best individual and fitness value
        self.best_individual = best_individual
        self.best_fitness = best_fitness
        # Update the budget
        self.budget -= len(self.population)
        # Check if the budget is exhausted
        if self.budget <= 0:
            # If the budget is exhausted, return the best individual found so far
            return self.best_individual, self.best_fitness
        # Return the best individual found so far
        return best_individual, best_fitness

# One-line description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 