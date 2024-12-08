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
            offspring = []
            for _ in range(self.pop_size):
                if random.random() < self.crossover_prob:
                    parent1, parent2 = random.sample(parents, 2)
                    crossover_point = random.randint(0, self.dim - 1)
                    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                    offspring.append(child1)
                    offspring.append(child2)
            offspring = np.array(offspring)

            # Mutate
            mutated_population = offspring.copy()
            for i in range(self.pop_size):
                if random.random() < self.mutate_prob:
                    mutated_population[i] += np.random.normal(0, 1)
                    mutated_population[i] = np.clip(mutated_population[i], -5.0, 5.0)
            mutated_population = np.array(mutated_population)

            # Evaluate offspring
            new_fitness = np.array([func(x) for x in mutated_population])

            # Replace worst individuals
            sorted_indices = np.argsort(new_fitness)
            population = np.concatenate((population[sorted_indices[:int(0.2 * self.pop_size)]], mutated_population))

            # Update fitness
            fitness = np.concatenate((fitness, new_fitness))

        # Return best individual
        return population[np.argmin(fitness)]

    def select_parents(self, population, fitness):
        # Select top 20% of population with highest fitness
        sorted_indices = np.argsort(fitness)
        return population[sorted_indices[:int(0.2 * self.pop_size)]]

# Example usage:
def func(x):
    return np.sum(x**2)

optimizer = HyperEvolutionOptimizer(100, 10)
best_individual = optimizer(func)
print(best_individual)