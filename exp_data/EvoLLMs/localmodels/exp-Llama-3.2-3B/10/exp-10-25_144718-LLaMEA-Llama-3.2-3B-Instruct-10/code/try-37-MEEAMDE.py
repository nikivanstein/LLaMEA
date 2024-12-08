import numpy as np
import random

class MEEAMDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.5
        self.differential_evolution_rate = 0.2
        self.fitness_values = []
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of each point in the population
            fitness = np.array([func(x) for x in self.population])

            # Sort the population based on fitness
            idx = np.argsort(fitness)
            self.population = self.population[idx]

            # Select the best points for the next generation
            next_generation = self.population[:int(self.population_size * 0.2)]

            # Perform crossover and mutation
            for i in range(self.population_size):
                if random.random() < self.crossover_rate:
                    parent1, parent2 = random.sample(next_generation, 2)
                    child = np.mean([parent1, parent2], axis=0)
                    if random.random() < self.mutation_rate:
                        child += np.random.uniform(-0.1, 0.1, self.dim)
                else:
                    child = next_generation[i]

                # Evaluate the fitness of the child
                fitness_child = func(child)
                idx_child = np.where(fitness == fitness_child)[0][0]
                next_generation[idx_child] = child

                # Perform differential evolution
                if random.random() < self.differential_evolution_rate:
                    target_idx = random.randint(0, self.population_size - 1)
                    target = next_generation[target_idx]
                    for j in range(self.dim):
                        next_generation[idx_child, j] += target[j] - next_generation[target_idx, j]

            # Update the population
            self.population = np.array(next_generation)

            # Sort the population based on fitness
            idx = np.argsort(fitness)
            self.population = self.population[idx]

        # Return the best point in the population
        best_point = self.population[0]
        return best_point

# Example usage:
def func(x):
    return np.sum(x**2)

meeamde = MEEAMDE(budget=100, dim=10)
best_point = meeamde(func)
print(best_point)