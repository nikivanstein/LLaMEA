import numpy as np
import random
import time
import copy

class DifferentialEvolutionWithHybridEvolutionStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.crossover_probability = 0.9
        self.mutation_probability = 0.1
        self.fitness_history = []
        self.best_individuals = []
        self.hybridization_rate = 0.45

    def __call__(self, func):
        start_time = time.time()
        for i in range(self.budget):
            # Initialize population with random points
            population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

            # Evaluate fitness for each individual
            fitness = np.array([func(individual) for individual in population])

            # Evolutionary strategy
            for _ in range(100):  # 100 generations
                # Select parents using differential evolution
                parents = []
                for j in range(self.population_size):
                    individual = population[j]
                    target = population[(np.random.randint(0, self.population_size) + j) % self.population_size]
                    differential = target - individual
                    mutant = individual + differential * np.random.uniform(-1.0, 1.0, size=self.dim)
                    parents.append(mutant)

                # Hybridize with evolutionary strategy
                hybridized_parents = []
                for j in range(self.population_size):
                    if random.random() < self.hybridization_rate:
                        individual = parents[j]
                        mutant = population[np.random.randint(0, self.population_size)]
                        hybridized_parent = (individual + mutant) / 2
                        hybridized_parents.append(hybridized_parent)
                    else:
                        hybridized_parents.append(parents[j])

                # Crossover
                offspring = []
                for j in range(self.population_size):
                    if random.random() < self.crossover_probability:
                        parent1, parent2 = random.sample(hybridized_parents, 2)
                        child = (parent1 + parent2) / 2
                        offspring.append(child)
                    else:
                        offspring.append(hybridized_parents[np.random.randint(0, len(hybridized_parents))])

                # Mutation
                for j in range(self.population_size):
                    if random.random() < self.mutation_probability:
                        mutation = np.random.uniform(-1.0, 1.0, size=self.dim)
                        offspring[j] += mutation

                # Replace worst individual
                offspring = np.sort(offspring, axis=0)
                population = np.delete(population, np.argmin(fitness), axis=0)
                population = np.vstack((population, offspring))

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

es = DifferentialEvolutionWithHybridEvolutionStrategy(budget=100, dim=10)
best_fitness, best_individual = es(func)
print(f"Best fitness: {best_fitness}")
print(f"Best individual: {best_individual}")