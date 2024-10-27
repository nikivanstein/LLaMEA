import numpy as np
import random
import operator

class HyperEvolutionOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.mutate_prob = 0.1
        self.crossover_prob = 0.8
        self.fitness_func = None
        self.population = None

    def __call__(self, func):
        # Initialize population
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

        # Evaluate initial population
        fitness = np.array([func(x) for x in self.population])

        # Hyper-Evolutionary loop
        for i in range(self.budget):
            # Select parents
            parents = self.select_parents(fitness)

            # Crossover
            offspring = self.crossover(parents, self.mutate_prob)

            # Mutate
            offspring = self.mutate(offspring)

            # Evaluate offspring
            new_fitness = np.array([func(x) for x in offspring])

            # Replace worst individuals
            self.replace_worst(fitness, new_fitness)

            # Update fitness
            fitness = np.concatenate((fitness, new_fitness))

        # Return best individual
        return self.population[np.argmin(fitness)]

    def select_parents(self, fitness):
        # Select top 20% of population with highest fitness
        sorted_indices = np.argsort(fitness)
        return self.population[sorted_indices[:int(0.2 * self.pop_size)]]


    def crossover(self, parents, mutate_prob):
        # Perform single-point crossover
        offspring = []
        for _ in range(self.pop_size):
            parent1, parent2 = random.sample(parents, 2)
            crossover_point = random.randint(0, self.dim - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            if np.random.rand() < mutate_prob:
                child1 += np.random.normal(0, 1)
                child1 = np.clip(child1, -5.0, 5.0)
                child2 += np.random.normal(0, 1)
                child2 = np.clip(child2, -5.0, 5.0)
            offspring.append(child1)
            offspring.append(child2)
        return np.array(offspring)

    def mutate(self, population):
        # Perform Gaussian mutation
        mutated_population = population.copy()
        for i in range(self.pop_size):
            if np.random.rand() < self.mutate_prob:
                mutated_population[i] += np.random.normal(0, 1)
                mutated_population[i] = np.clip(mutated_population[i], -5.0, 5.0)
        return mutated_population

    def replace_worst(self, fitness, new_fitness):
        # Replace worst 20% of population with offspring
        sorted_indices = np.argsort(new_fitness)
        self.population = np.concatenate((self.population[sorted_indices[:int(0.2 * self.pop_size)]], np.array([x for x in offspring if x not in self.population])))

# ```