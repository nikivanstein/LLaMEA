import numpy as np
from scipy.optimize import differential_evolution
import random

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.population_size = 20
        self.crossover_prob = 0.4
        self.mutation_prob = 0.1

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        population = [np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim) for _ in range(self.population_size)]

        for _ in range(self.budget):
            fitness = [func(individual) for individual in population]
            min_fitness_idx = np.argmin(fitness)
            min_fitness_individual = population[min_fitness_idx]

            new_population = []
            for individual in population:
                if random.random() < self.crossover_prob:
                    new_individual = self.crossover(individual, min_fitness_individual)
                else:
                    new_individual = individual
                if random.random() < self.mutation_prob:
                    new_individual = self.mutate(new_individual)
                new_population.append(new_individual)

            population = new_population

        min_fitness_individual = min(population, key=func)
        return func(min_fitness_individual), min_fitness_individual

    def crossover(self, parent1, parent2):
        child = [0] * self.dim
        for i in range(self.dim):
            if random.random() < self.crossover_prob:
                child[i] = random.choice([parent1[i], parent2[i]])
            else:
                child[i] = parent1[i]
        return child

    def mutate(self, individual):
        mutated_individual = individual.copy()
        for i in range(self.dim):
            if random.random() < self.mutation_prob:
                mutated_individual[i] += random.uniform(-1, 1)
                mutated_individual[i] = max(self.bounds[i][0], min(self.bounds[i][1], mutated_individual[i]))
        return mutated_individual

# Example usage
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return x[0]**2 + 2*x[1]**2

def f3(x):
    return x[0]**2 + 3*x[1]**2

def f4(x):
    return x[0]**2 + 4*x[1]**2

def f5(x):
    return x[0]**2 + 5*x[1]**2

def f6(x):
    return x[0]**2 + 6*x[1]**2

def f7(x):
    return x[0]**2 + 7*x[1]**2

def f8(x):
    return x[0]**2 + 8*x[1]**2

def f9(x):
    return x[0]**2 + 9*x[1]**2

def f10(x):
    return x[0]**2 + 10*x[1]**2

def f11(x):
    return x[0]**2 + 11*x[1]**2

def f12(x):
    return x[0]**2 + 12*x[1]**2

def f13(x):
    return x[0]**2 + 13*x[1]**2

def f14(x):
    return x[0]**2 + 14*x[1]**2

def f15(x):
    return x[0]**2 + 15*x[1]**2

def f16(x):
    return x[0]**2 + 16*x[1]**2

def f17(x):
    return x[0]**2 + 17*x[1]**2

def f18(x):
    return x[0]**2 + 18*x[1]**2

def f19(x):
    return x[0]**2 + 19*x[1]**2

def f20(x):
    return x[0]**2 + 20*x[1]**2

def f21(x):
    return x[0]**2 + 21*x[1]**2

def f22(x):
    return x[0]**2 + 22*x[1]**2

def f23(x):
    return x[0]**2 + 23*x[1]**2

def f24(x):
    return x[0]**2 + 24*x[1]**2

budget = 100
dim = 2
algorithm = HybridEvolutionaryAlgorithm(budget, dim)
for func in [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24]:
    result = algorithm(func)
    print(f"Function: {func.__name__}, Result: {result[0]}, Individual: {result[1]}")