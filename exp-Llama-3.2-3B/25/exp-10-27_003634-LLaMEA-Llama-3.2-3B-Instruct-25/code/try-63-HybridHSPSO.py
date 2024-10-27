import numpy as np
import random

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
        self.probability = 0.25

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

            # Refine the strategy with probability 0.25
            if np.random.rand() < self.probability:
                refined_x = self.refine_strategy(new_x)
                new_x = np.concatenate((new_x, refined_x))

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

            # Check if the optimization is complete
            if np.all(fitness < 1e-6):
                break

    def refine_strategy(self, x):
        refined_x = []
        for i in range(len(x)):
            if np.random.rand() < self.probability:
                # Perform adaptive crossover and mutation
                parent1 = self.x[np.random.randint(0, len(self.x))]
                parent2 = self.x[np.random.randint(0, len(self.x))]
                child = self.adaptive_crossover(parent1, parent2)
                child += np.random.uniform(-0.1, 0.1, self.dim)
                refined_x.append(child)
        return np.array(refined_x)

    def adaptive_crossover(self, parent1, parent2):
        child = np.zeros(self.dim)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child

# Example usage
def noiseless_func(x):
    return np.sum(x**2)

hybridHSPSO = HybridHSPSO(budget=100, dim=10)
hybridHSPSO noiseless_func