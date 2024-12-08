import numpy as np
import random

class HyperEvolutionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.mutate_prob = 0.1
        self.crossover_prob = 0.8
        self.fitness_func = None
        self.probability = 0.2

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

        # Evaluate initial population
        fitness = np.array([func(x) for x in population])

        # Hyper-Evolutionary loop
        for i in range(self.budget):
            # Select parents
            parents = self.select_parents(population, fitness)

            # Crossover
            offspring = self.crossover(parents)

            # Mutate
            offspring = self.mutate(offspring, self.mutate_prob)

            # Evaluate offspring
            new_fitness = np.array([func(x) for x in offspring])

            # Replace worst individuals
            population = self.replace_worst(population, offspring, new_fitness, self.probability)

            # Update fitness
            fitness = np.concatenate((fitness, new_fitness))

        # Return best individual
        return population[np.argmin(fitness)]

    def select_parents(self, population, fitness):
        # Select top 20% of population with highest fitness
        sorted_indices = np.argsort(fitness)
        return population[sorted_indices[:int(0.2 * self.pop_size)]]

    def crossover(self, parents):
        # Perform single-point crossover
        offspring = []
        for _ in range(self.pop_size):
            parent1, parent2 = random.sample(parents, 2)
            crossover_point = random.randint(0, self.dim - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            offspring.append(child1)
            offspring.append(child2)
        return np.array(offspring)

    def mutate(self, population, mutate_prob):
        # Perform Gaussian mutation
        mutated_population = population.copy()
        for i in range(self.pop_size):
            if random.random() < mutate_prob:
                mutated_population[i] += np.random.normal(0, 1)
                mutated_population[i] = np.clip(mutated_population[i], -5.0, 5.0)
        return mutated_population

    def replace_worst(self, population, offspring, new_fitness, probability):
        # Replace worst individuals with offspring with a certain probability
        sorted_indices = np.argsort(new_fitness)
        replace_indices = sorted_indices[:int(0.2 * self.pop_size)]
        replace_indices = np.random.choice(replace_indices, size=int(0.2 * self.pop_size * probability), replace=True)
        return np.concatenate((population[sorted_indices[replace_indices]], offspring))
