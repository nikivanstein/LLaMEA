import numpy as np
from scipy.optimize import differential_evolution
import random

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.population_size = 10
        self.population = [np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim) for _ in range(self.population_size)]

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        for _ in range(self.budget):
            # Select parents using tournament selection
            parents = []
            for _ in range(self.population_size):
                tournament = random.sample(self.population, 3)
                winner = min(tournament, key=lambda x: func(x))
                parents.append(winner)

            # Perform crossover and mutation
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = random.sample(parents, 2)
                child = parent1 + (parent2 - parent1) * 0.4
                child = np.clip(child, self.bounds[0][0], self.bounds[0][1])
                new_population.append(child)

            # Replace the old population with the new one
            self.population = new_population

            # Evaluate the fitness of the new population
            fitness = [func(individual) for individual in self.population]
            min_fitness, min_individual = min(zip(fitness, self.population), key=lambda x: x[0])
            self.population = [individual for individual, fitness in zip(self.population, fitness) if fitness > min_fitness]

        # Return the best individual
        min_fitness, min_individual = min(zip([func(individual) for individual in self.population], self.population), key=lambda x: x[0])
        return min_individual, min_fitness

# Example usage:
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return x[0]**2 + 2*x[1]**2

def f3(x):
    return x[0]**3 + 3*x[1]**3

def f4(x):
    return x[0]**4 + 4*x[1]**4

def f5(x):
    return x[0]**5 + 5*x[1]**5

def f6(x):
    return x[0]**6 + 6*x[1]**6

def f7(x):
    return x[0]**7 + 7*x[1]**7

def f8(x):
    return x[0]**8 + 8*x[1]**8

def f9(x):
    return x[0]**9 + 9*x[1]**9

def f10(x):
    return x[0]**10 + 10*x[1]**10

def f11(x):
    return x[0]**11 + 11*x[1]**11

def f12(x):
    return x[0]**12 + 12*x[1]**12

def f13(x):
    return x[0]**13 + 13*x[1]**13

def f14(x):
    return x[0]**14 + 14*x[1]**14

def f15(x):
    return x[0]**15 + 15*x[1]**15

def f16(x):
    return x[0]**16 + 16*x[1]**16

def f17(x):
    return x[0]**17 + 17*x[1]**17

def f18(x):
    return x[0]**18 + 18*x[1]**18

def f19(x):
    return x[0]**19 + 19*x[1]**19

def f20(x):
    return x[0]**20 + 20*x[1]**20

def f21(x):
    return x[0]**21 + 21*x[1]**21

def f22(x):
    return x[0]**22 + 22*x[1]**22

def f23(x):
    return x[0]**23 + 23*x[1]**23

def f24(x):
    return x[0]**24 + 24*x[1]**24

# Create an instance of the algorithm
algorithm = HybridEvolutionaryAlgorithm(budget=100, dim=2)

# Optimize a function
best_individual, best_fitness = algorithm(f1)

print("Best individual:", best_individual)
print("Best fitness:", best_fitness)