import numpy as np
import random
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        # Initialize population
        population = [np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim) for _ in range(100)]

        # Initialize logger
        logger = {}

        # Evolve population
        for i in range(self.budget):
            # Evaluate fitness
            fitness = [func(individual) for individual in population]

            # Select parents
            parents = np.array(population)[np.argsort(fitness)]

            # Crossover
            offspring = []
            for _ in range(20):
                parent1, parent2 = parents[np.random.randint(0, len(parents)-1)], parents[np.random.randint(0, len(parents)-1)]
                child = (parent1 + parent2) / 2
                offspring.append(child)

            # Mutation
            for child in offspring:
                if random.random() < 0.4:
                    child += np.random.uniform(-1, 1, self.dim)

            # Replace worst individual
            population = np.array(population)[np.argsort(fitness)]
            population = np.concatenate((offspring, population))

            # Update logger
            for individual in population:
                logger[individual] = fitness[np.where(population == individual)[0][0]]

        # Return best individual
        best_individual = population[np.argmin([func(individual) for individual in population])]
        return func(best_individual), best_individual

# Test the algorithm
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2

def f3(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2

def f4(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f5(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f6(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f7(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f8(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f9(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f10(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f11(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f12(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f13(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f14(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f15(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f16(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f17(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f18(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f19(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f20(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f21(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f22(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f23(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

def f24(x):
    return x[0]**2 + x[1]**2 + 0.5 * (x[0] + x[1])**2 + 0.1 * (x[0] - x[1])**2 + 0.01 * (x[0]**2 + x[1]**2)

# Test the algorithm
budget = 100
dim = 2
algorithm = HybridEvolutionaryAlgorithm(budget, dim)
best_function, best_individual = algorithm(f1)
print(f"Best function: {best_function}, Best individual: {best_individual}")