import numpy as np
import random
from scipy.optimize import differential_evolution

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.population_size = 20
        self.population = [self.x0 + np.random.uniform(-1, 1, self.dim) for _ in range(self.population_size)]
        self.strategy_refinement_probability = 0.4

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        for _ in range(self.budget):
            best_individual = self.f(func)
            self.population = [self.evaluate_fitness(individual) for individual in self.population]
            self.population.sort()
            self.population = self.population[:self.population_size]

            if random.random() < self.strategy_refinement_probability:
                new_individual = random.choice(self.population)
                new_individual = new_individual + np.random.uniform(-1, 1, self.dim)
                self.population = [self.evaluate_fitness(individual) for individual in self.population]
                self.population.sort()
                self.population = self.population[:self.population_size]

            best_individual = self.f(func)
            self.population = [self.evaluate_fitness(individual) for individual in self.population]
            self.population.sort()
            self.population = self.population[:self.population_size]

        return best_individual

    def f(self, func):
        return min(func(individual) for individual in self.population)

    def evaluate_fitness(self, individual):
        fitness = func(individual)
        if fitness < self.f(func):
            self.population = [self.evaluate_fitness(individual) for individual in self.population]
            self.population.sort()
            self.population = self.population[:self.population_size]
        return fitness

# Usage
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1]

def f3(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f4(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f5(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f6(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f7(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f8(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f9(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f10(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f11(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f12(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f13(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f14(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f15(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f16(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f17(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f18(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f19(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f20(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f21(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f22(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f23(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

def f24(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0] + 0.5*x[1] + 1e-5

# Usage
budget = 100
dim = 2
algorithm = HybridEvolutionaryAlgorithm(budget, dim)
best_individual, _ = algorithm(f1)
print(best_individual)