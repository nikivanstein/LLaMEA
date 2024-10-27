import numpy as np
import random
import copy

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        def crossover(parent1, parent2):
            child = copy.deepcopy(parent1)
            for i in range(self.dim):
                if random.random() < 0.4:
                    child[i] = parent2[i]
            return child

        def mutate(individual):
            for i in range(self.dim):
                if random.random() < 0.4:
                    individual[i] += random.uniform(-1.0, 1.0)
                    individual[i] = max(self.bounds[i][0], min(individual[i], self.bounds[i][1]))
            return individual

        def evaluate_fitness(individual):
            return func(individual)

        population = [self.x0]
        for _ in range(self.budget - 1):
            parent1, parent2 = sorted(population, key=evaluate_fitness)[0], sorted(population, key=evaluate_fitness)[1]
            child = crossover(parent1, parent2)
            child = mutate(child)
            population.append(child)

        best_individual = min(population, key=evaluate_fitness)
        return evaluate_fitness(best_individual), best_individual

# Example usage:
def f1(x):
    return sum(x**2)

def f2(x):
    return sum(x**2) + 1

def f3(x):
    return sum(x**2) + 1e-5

def f4(x):
    return sum(x**2) + 1e-5 + np.sin(x)

def f5(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x)

def f6(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x)

def f7(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f8(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f9(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f10(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f11(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f12(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f13(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f14(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f15(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f16(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f17(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f18(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f19(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f20(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f21(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f22(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f23(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

def f24(x):
    return sum(x**2) + 1e-5 + np.sin(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x) * np.cos(x)

hybrid_eva = HybridEvolutionaryAlgorithm(budget=10, dim=10)
best_fitness, best_individual = hybrid_eva(f1)
print(f"Best fitness: {best_fitness}, Best individual: {best_individual}")