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
            # Select top 20% of population with highest fitness
            sorted_indices = np.argsort(fitness)
            parents = population[sorted_indices[:int(0.2 * self.pop_size)]]

            # Perform single-point crossover
            offspring = []
            for _ in range(self.pop_size):
                parent1, parent2 = random.sample(parents, 2)
                crossover_point = random.randint(0, self.dim - 1)
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                offspring.append(child1)
                offspring.append(child2)
            offspring = np.array(offspring)

            # Perform Gaussian mutation
            mutated_offspring = offspring.copy()
            for i in range(self.pop_size):
                if random.random() < self.mutate_prob:
                    mutated_offspring[i] += np.random.normal(0, 1)
                    mutated_offspring[i] = np.clip(mutated_offspring[i], -5.0, 5.0)

            # Evaluate offspring
            new_fitness = np.array([func(x) for x in mutated_offspring])

            # Replace worst individuals
            sorted_indices = np.argsort(new_fitness)
            population = np.concatenate((population[sorted_indices[int(0.2 * self.pop_size):]], mutated_offspring))

        # Return best individual
        return population[np.argmin(fitness)]

# Example usage
def func(x):
    return np.sum(x**2)

optimizer = HyperEvolutionOptimizer(100, 10)
best_individual = optimizer(func)
print(best_individual)