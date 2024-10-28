import numpy as np
from scipy.optimize import differential_evolution
import random

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.population_size = 100
        self.crossover_probability = 0.4
        self.mutation_probability = 0.1
        self.logger = []

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        if len(self.logger) == 0:
            self.logger.append(func(self.x0))

        while self.budget > 0:
            self.budget -= 1

            # Select parents
            parents = np.array([random.choice(self.logger) for _ in range(self.population_size)])

            # Crossover
            offspring = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                if random.random() < self.crossover_probability:
                    child = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
                    child = child * (parent1 - parent2) + parent2
                    offspring.append(child)
                else:
                    offspring.append(parent1)

            # Mutation
            for i in range(self.population_size):
                if random.random() < self.mutation_probability:
                    offspring[i] = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

            # Evaluate fitness
            offspring_fitness = [func(individual) for individual in offspring]

            # Replace worst individual
            worst_index = np.argmin(offspring_fitness)
            self.logger[worst_index] = offspring[worst_index]

            # Update logger
            self.logger.sort()
            self.logger = self.logger[:self.population_size]

        # Return best individual
        best_individual = np.min(self.logger)
        return best_individual, self.logger[0]

# Example usage:
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return x[0]**2 + x[1]**2 + 1e-6

def f3(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6

def f4(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6

def f5(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f6(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f7(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f8(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f9(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f10(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f11(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f12(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f13(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f14(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f15(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f16(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f17(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f18(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f19(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f20(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f21(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f22(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f23(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

def f24(x):
    return x[0]**2 + x[1]**2 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6 + 1e-6

# Example usage
algorithm = HybridEvolutionaryAlgorithm(budget=10, dim=2)
best_individual, best_fitness = algorithm(f1)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f2)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f3)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f4)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f5)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f6)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f7)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f8)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f9)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f10)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f11)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f12)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f13)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f14)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f15)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f16)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f17)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f18)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f19)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f20)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f21)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f22)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f23)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)

best_individual, best_fitness = algorithm(f24)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)