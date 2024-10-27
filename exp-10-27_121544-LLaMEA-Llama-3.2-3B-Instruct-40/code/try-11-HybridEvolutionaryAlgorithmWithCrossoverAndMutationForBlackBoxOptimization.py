import numpy as np
import random
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithmWithCrossoverAndMutationForBlackBoxOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.population_size = 100
        self.mutation_rate = 0.1

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        population = [np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim) for _ in range(self.population_size)]

        for _ in range(self.budget):
            # Evaluate the fitness of each individual
            fitness = [func(individual) for individual in population]

            # Select the fittest individuals
            fittest_individuals = np.argsort(fitness)[:self.population_size // 2]

            # Create a new generation by crossover and mutation
            new_generation = []
            for _ in range(self.population_size):
                if random.random() < 0.4:
                    parent1, parent2 = random.sample(fittest_individuals, 2)
                    child = (population[parent1] + population[parent2]) / 2
                    if random.random() < self.mutation_rate:
                        child += np.random.uniform(-0.1, 0.1, self.dim)
                else:
                    child = population[fittest_individuals[np.random.randint(0, self.population_size // 2)]]
                new_generation.append(child)

            # Replace the old population with the new generation
            population = new_generation

        # Return the best individual
        return min(population, key=func)

# Usage
def f1(x):
    return sum(x**2)

def f2(x):
    return sum(x**2) + 0.1 * np.sum(x**2)

def f3(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2)

def f4(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2)

def f5(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2)

def f6(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2)

def f7(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2)

def f8(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2)

def f9(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2) + 0.00000001 * np.sum(x**2)

def f10(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2) + 0.00000001 * np.sum(x**2) + 0.000000001 * np.sum(x**2)

def f11(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2) + 0.00000001 * np.sum(x**2) + 0.000000001 * np.sum(x**2) + 0.0000000001 * np.sum(x**2)

def f12(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2) + 0.00000001 * np.sum(x**2) + 0.000000001 * np.sum(x**2) + 0.0000000001 * np.sum(x**2) + 0.00000000001 * np.sum(x**2)

def f13(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2) + 0.00000001 * np.sum(x**2) + 0.000000001 * np.sum(x**2) + 0.0000000001 * np.sum(x**2) + 0.00000000001 * np.sum(x**2) + 0.000000000001 * np.sum(x**2)

def f14(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2) + 0.00000001 * np.sum(x**2) + 0.000000001 * np.sum(x**2) + 0.0000000001 * np.sum(x**2) + 0.00000000001 * np.sum(x**2) + 0.000000000001 * np.sum(x**2) + 0.0000000000001 * np.sum(x**2)

def f15(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2) + 0.00000001 * np.sum(x**2) + 0.000000001 * np.sum(x**2) + 0.0000000001 * np.sum(x**2) + 0.00000000001 * np.sum(x**2) + 0.000000000001 * np.sum(x**2) + 0.0000000000001 * np.sum(x**2) + 0.00000000000001 * np.sum(x**2)

def f16(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2) + 0.00000001 * np.sum(x**2) + 0.000000001 * np.sum(x**2) + 0.0000000001 * np.sum(x**2) + 0.00000000001 * np.sum(x**2) + 0.000000000001 * np.sum(x**2) + 0.0000000000001 * np.sum(x**2) + 0.00000000000001 * np.sum(x**2) + 0.000000000000001 * np.sum(x**2)

def f17(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2) + 0.00000001 * np.sum(x**2) + 0.000000001 * np.sum(x**2) + 0.0000000001 * np.sum(x**2) + 0.00000000001 * np.sum(x**2) + 0.000000000001 * np.sum(x**2) + 0.0000000000001 * np.sum(x**2) + 0.00000000000001 * np.sum(x**2) + 0.000000000000001 * np.sum(x**2) + 0.0000000000000001 * np.sum(x**2)

def f18(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2) + 0.00000001 * np.sum(x**2) + 0.000000001 * np.sum(x**2) + 0.0000000001 * np.sum(x**2) + 0.00000000001 * np.sum(x**2) + 0.000000000001 * np.sum(x**2) + 0.0000000000001 * np.sum(x**2) + 0.00000000000001 * np.sum(x**2) + 0.000000000000001 * np.sum(x**2) + 0.0000000000000001 * np.sum(x**2)

def f19(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2) + 0.00000001 * np.sum(x**2) + 0.000000001 * np.sum(x**2) + 0.0000000001 * np.sum(x**2) + 0.00000000001 * np.sum(x**2) + 0.000000000001 * np.sum(x**2) + 0.0000000000001 * np.sum(x**2) + 0.00000000000001 * np.sum(x**2) + 0.000000000000001 * np.sum(x**2) + 0.0000000000000001 * np.sum(x**2)

def f20(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2) + 0.00000001 * np.sum(x**2) + 0.000000001 * np.sum(x**2) + 0.0000000001 * np.sum(x**2) + 0.00000000001 * np.sum(x**2) + 0.000000000001 * np.sum(x**2) + 0.0000000000001 * np.sum(x**2) + 0.00000000000001 * np.sum(x**2) + 0.000000000000001 * np.sum(x**2) + 0.0000000000000001 * np.sum(x**2)

def f21(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2) + 0.00000001 * np.sum(x**2) + 0.000000001 * np.sum(x**2) + 0.0000000001 * np.sum(x**2) + 0.00000000001 * np.sum(x**2) + 0.000000000001 * np.sum(x**2) + 0.0000000000001 * np.sum(x**2) + 0.00000000000001 * np.sum(x**2) + 0.000000000000001 * np.sum(x**2) + 0.0000000000000001 * np.sum(x**2)

def f22(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2) + 0.00000001 * np.sum(x**2) + 0.000000001 * np.sum(x**2) + 0.0000000001 * np.sum(x**2) + 0.00000000001 * np.sum(x**2) + 0.000000000001 * np.sum(x**2) + 0.0000000000001 * np.sum(x**2) + 0.00000000000001 * np.sum(x**2) + 0.000000000000001 * np.sum(x**2) + 0.0000000000000001 * np.sum(x**2)

def f23(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2) + 0.00000001 * np.sum(x**2) + 0.000000001 * np.sum(x**2) + 0.0000000001 * np.sum(x**2) + 0.00000000001 * np.sum(x**2) + 0.000000000001 * np.sum(x**2) + 0.0000000000001 * np.sum(x**2) + 0.00000000000001 * np.sum(x**2) + 0.000000000000001 * np.sum(x**2) + 0.0000000000000001 * np.sum(x**2)

def f24(x):
    return sum(x**2) + 0.1 * np.sum(x**2) + 0.01 * np.sum(x**2) + 0.001 * np.sum(x**2) + 0.0001 * np.sum(x**2) + 0.00001 * np.sum(x**2) + 0.000001 * np.sum(x**2) + 0.0000001 * np.sum(x**2) + 0.00000001 * np.sum(x**2) + 0.000000001 * np.sum(x**2) + 0.0000000001 * np.sum(x**2) + 0.00000000001 * np.sum(x**2) + 0.000000000001 * np.sum(x**2) + 0.0000000000001 * np.sum(x**2) + 0.00000000000001 * np.sum(x**2) + 0.000000000000001 * np.sum(x**2) + 0.0000000000000001 * np.sum(x**2)

# Usage
algorithm = HybridEvolutionaryAlgorithmWithCrossoverAndMutationForBlackBoxOptimization(budget=100, dim=10)
result = algorithm(f1)
print(result)