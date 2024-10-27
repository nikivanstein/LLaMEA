import numpy as np
import random
import functools

class HybridHSPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros(self.dim)
        self.x = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of each individual in the population
            fitness = func(self.x)

            # Update the pbest and gbest
            for i in range(self.population_size):
                if np.any(fitness[i] < fitness[self.pbest[i, :]]):
                    self.pbest[i, :] = self.x[i, :]
                if np.any(fitness[i] < fitness[self.gbest]):
                    self.gbest = self.x[i, :]

            # Select the fittest individuals for crossover and mutation
            fittest_indices = np.argsort(fitness)[:, ::-1][:, :int(self.population_size * self.crossover_rate)]
            fittest_x = self.x[fittest_indices]

            # Perform crossover and mutation
            new_x = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                if np.random.rand() < self.crossover_rate:
                    new_x[i, :] = fittest_x[np.random.randint(0, len(fittest_x)), :]
                else:
                    new_x[i, :] = self.x[i, :]
                if np.random.rand() < self.mutation_rate:
                    new_x[i, :] += np.random.uniform(-0.1, 0.1, self.dim)

            # Ensure the search space bounds
            new_x = np.clip(new_x, -5.0, 5.0, out=new_x)

            # Replace the old population with the new one
            self.x = new_x

            # Evaluate the fitness of the new population
            fitness = func(self.x)

            # Update the pbest and gbest
            for i in range(self.population_size):
                if np.any(fitness[i] < fitness[self.pbest[i, :]]):
                    self.pbest[i, :] = self.x[i, :]
                if np.any(fitness[i] < fitness[self.gbest]):
                    self.gbest = self.x[i, :]

            # Refine the strategy with probability 0.25
            if random.random() < 0.25:
                self.refine_strategy()

            # Check if the optimization is complete
            if np.all(fitness < 1e-6):
                break

    def refine_strategy(self):
        for i in range(self.population_size):
            # Select a random individual to refine
            individual = self.x[i, :]
            # Refine the individual by changing 25% of its lines
            refined_individual = individual.copy()
            for j in range(self.dim):
                if np.random.rand() < 0.25:
                    refined_individual[j] += np.random.uniform(-0.1, 0.1)
            # Ensure the search space bounds
            refined_individual = np.clip(refined_individual, -5.0, 5.0, out=refined_individual)
            # Replace the old individual with the refined one
            self.x[i, :] = refined_individual

# Example usage
def noiseless_func(x):
    return np.sum(x**2)

hybridHSPSO = HybridHSPSO(budget=100, dim=10)
hybridHSPSO noiseless_func