import numpy as np
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.population_size = 10
        self.mutation_rate = 0.1

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        population = np.random.uniform(self.bounds[0][0], self.bounds[0][1], (self.population_size, self.dim))
        for _ in range(self.budget):
            new_individuals = []
            for individual in population:
                if np.random.rand() < self.mutation_rate:
                    new_individual = individual + np.random.uniform(-1, 1, self.dim)
                    new_individuals.append(new_individual)
                else:
                    new_individuals.append(individual)
            population = np.array(new_individuals)

        fitness_values = [func(individual) for individual in population]
        indices = np.argsort(fitness_values)
        population = population[indices]
        fitness_values = fitness_values[indices]

        new_population = population[:self.population_size//2]
        for _ in range(self.population_size//2):
            parent1, parent2 = np.random.choice(population, size=2, replace=False)
            child = (parent1 + parent2) / 2
            new_population.append(child)

        return np.min(fitness_values), np.min([func(individual) for individual in new_population])

# Example usage:
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return x[0]**2 + x[1]**2 + 1.5 * np.sin(2 * np.pi * x[0]) + 0.5 * np.sin(4 * np.pi * x[1])

def f3(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 1 * np.sin(4 * np.pi * x[1])

def f4(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f5(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f6(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f7(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f8(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f9(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f10(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f11(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f12(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f13(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f14(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f15(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f16(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f17(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f18(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f19(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f20(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f21(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f22(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f23(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

def f24(x):
    return x[0]**2 + x[1]**2 + 2 * np.sin(2 * np.pi * x[0]) + 2 * np.sin(4 * np.pi * x[1])

# Initialize the algorithm
algorithm = HybridEvolutionaryAlgorithm(budget=10, dim=2)

# Evaluate the function
result = algorithm(f1)

print(result)