import numpy as np
import random
import time

class DifferentialEvolutionWithAdaptiveProbability:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.crossover_probability = 0.9
        self.mutation_probability = 0.1
        self.adaptive_probability = 0.45
        self.fitness_history = []
        self.best_individuals = []
        self.differential_evolution_matrix = np.zeros((self.population_size, self.population_size))

    def __call__(self, func):
        start_time = time.time()
        for i in range(self.budget):
            # Initialize population with random points
            population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

            # Evaluate fitness for each individual
            fitness = np.array([func(individual) for individual in population])

            # Differential evolution
            for j in range(self.population_size):
                # Select parents using differential evolution
                parent1 = population[np.random.randint(0, self.population_size)]
                parent2 = population[np.random.randint(0, self.population_size)]
                child = (parent1 + self.differential_evolution_matrix[j] * (parent2 - parent1))
                child = child + np.random.uniform(-1.0, 1.0, size=self.dim)

                # Crossover
                if random.random() < self.crossover_probability:
                    child = (child + parent1) / 2
                else:
                    child = parent2

                # Mutation
                if random.random() < self.mutation_probability:
                    mutation = np.random.uniform(-1.0, 1.0, size=self.dim)
                    child += mutation

                # Replace worst individual
                offspring = np.array([child])
                offspring = np.sort(offspring, axis=0)
                population = np.delete(population, np.argmin(fitness), axis=0)
                population = np.vstack((population, offspring))

                # Update differential evolution matrix
                self.differential_evolution_matrix[j] = parent2 - parent1

                # Evaluate fitness for each individual
                fitness = np.array([func(individual) for individual in population])

            # Update fitness history and best individuals
            self.fitness_history.append(fitness)
            self.best_individuals.append(np.argmin(fitness))

        end_time = time.time()
        print(f"Optimization time: {end_time - start_time} seconds")
        return np.min(self.fitness_history), self.best_individuals[-1]

# Example usage:
def func(x):
    return np.sum(x**2)

deap = DifferentialEvolutionWithAdaptiveProbability(budget=100, dim=10)
best_fitness, best_individual = deap(func)
print(f"Best fitness: {best_fitness}")
print(f"Best individual: {best_individual}")