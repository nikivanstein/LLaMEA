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
            offspring = self.crossover(parents)

            # Mutate
            mutated_population = []
            for individual in population:
                if random.random() < self.mutate_prob:
                    mutated_population.append(self.mutate(individual))
                else:
                    mutated_population.append(individual)
            mutated_population += offspring

            # Evaluate offspring
            new_fitness = np.array([func(x) for x in mutated_population])

            # Replace worst individuals
            sorted_indices = np.argsort(new_fitness)
            population = np.array(mutated_population)[sorted_indices[:int(0.2 * self.pop_size)]]

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

    def mutate(self, individual):
        # Perform Gaussian mutation
        mutated_individual = individual.copy()
        if random.random() < self.mutate_prob:
            mutated_individual += np.random.normal(0, 1)
            mutated_individual = np.clip(mutated_individual, -5.0, 5.0)
        return mutated_individual

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2 + x[2]**2

optimizer = HyperEvolutionOptimizer(100, 3)
best_individual = optimizer(func)
print(best_individual)