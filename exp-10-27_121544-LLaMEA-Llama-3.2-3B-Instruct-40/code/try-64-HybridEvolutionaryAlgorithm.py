import numpy as np
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

        population_size = 100
        mutation_rate = 0.2
        strategy_probabilities = [0.4, 0.3, 0.3]

        population = np.random.uniform(self.bounds[0][0], self.bounds[0][1], (population_size, self.dim))
        fitness = np.zeros(population_size)

        for i in range(population_size):
            individual = population[i]
            func_value = func(individual)

            if np.random.rand() < mutation_rate:
                new_individual = individual + np.random.uniform(-1, 1, self.dim)
                new_individual = np.clip(new_individual, self.bounds[0][0], self.bounds[0][1])
                fitness[i] = func(new_individual)

            else:
                fitness[i] = func_value

        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        # Refine strategy
        strategy = np.random.choice(['sbx', 'rand1', 'log-uniform'], p=strategy_probabilities)
        if strategy =='sbx':
            new_individual = best_individual + np.random.uniform(-1, 1, self.dim)
            new_individual = np.clip(new_individual, self.bounds[0][0], self.bounds[0][1])
        elif strategy == 'rand1':
            new_individual = best_individual + np.random.uniform(-1, 1, self.dim)
            new_individual = np.clip(new_individual, self.bounds[0][0], self.bounds[0][1])
        elif strategy == 'log-uniform':
            new_individual = best_individual + np.random.uniform(-1, 1, self.dim)
            new_individual = np.clip(new_individual, self.bounds[0][0], self.bounds[0][1])

        # Update population
        population = np.vstack((population, [new_individual]))
        fitness = np.append(fitness, [func(new_individual)])

        # Update best individual
        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        # Update budget
        self.budget -= 1

        return best_individual, best_fitness

# Test the algorithm
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return x[0]**2 + x[1]**2 + 1e-5

def f3(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8

def f4(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10

def f5(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12

def f6(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15

def f7(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18

def f8(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20

def f9(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20 + 1e-22

def f10(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20 + 1e-22 + 1e-24

def f11(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20 + 1e-22 + 1e-24 + 1e-26

def f12(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20 + 1e-22 + 1e-24 + 1e-26 + 1e-28

def f13(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20 + 1e-22 + 1e-24 + 1e-26 + 1e-28 + 1e-30

def f14(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20 + 1e-22 + 1e-24 + 1e-26 + 1e-28 + 1e-30 + 1e-32

def f15(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20 + 1e-22 + 1e-24 + 1e-26 + 1e-28 + 1e-30 + 1e-32 + 1e-34

def f16(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20 + 1e-22 + 1e-24 + 1e-26 + 1e-28 + 1e-30 + 1e-32 + 1e-34 + 1e-36

def f17(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20 + 1e-22 + 1e-24 + 1e-26 + 1e-28 + 1e-30 + 1e-32 + 1e-34 + 1e-36 + 1e-38

def f18(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20 + 1e-22 + 1e-24 + 1e-26 + 1e-28 + 1e-30 + 1e-32 + 1e-34 + 1e-36 + 1e-38 + 1e-40

def f19(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20 + 1e-22 + 1e-24 + 1e-26 + 1e-28 + 1e-30 + 1e-32 + 1e-34 + 1e-36 + 1e-38 + 1e-40 + 1e-42

def f20(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20 + 1e-22 + 1e-24 + 1e-26 + 1e-28 + 1e-30 + 1e-32 + 1e-34 + 1e-36 + 1e-38 + 1e-40 + 1e-42 + 1e-44

def f21(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20 + 1e-22 + 1e-24 + 1e-26 + 1e-28 + 1e-30 + 1e-32 + 1e-34 + 1e-36 + 1e-38 + 1e-40 + 1e-42 + 1e-44 + 1e-46

def f22(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20 + 1e-22 + 1e-24 + 1e-26 + 1e-28 + 1e-30 + 1e-32 + 1e-34 + 1e-36 + 1e-38 + 1e-40 + 1e-42 + 1e-44 + 1e-46 + 1e-48

def f23(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20 + 1e-22 + 1e-24 + 1e-26 + 1e-28 + 1e-30 + 1e-32 + 1e-34 + 1e-36 + 1e-38 + 1e-40 + 1e-42 + 1e-44 + 1e-46 + 1e-48 + 1e-50

def f24(x):
    return x[0]**2 + x[1]**2 + 1e-5 + 1e-8 + 1e-10 + 1e-12 + 1e-15 + 1e-18 + 1e-20 + 1e-22 + 1e-24 + 1e-26 + 1e-28 + 1e-30 + 1e-32 + 1e-34 + 1e-36 + 1e-38 + 1e-40 + 1e-42 + 1e-44 + 1e-46 + 1e-48 + 1e-50 + 1e-52

# Test the algorithm
if __name__ == '__main__':
    algorithm = HybridEvolutionaryAlgorithm(100, 2)
    func = f1
    individual, fitness = algorithm(func)
    print(individual, fitness)