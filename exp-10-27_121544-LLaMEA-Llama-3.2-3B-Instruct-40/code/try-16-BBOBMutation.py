import numpy as np
from scipy.optimize import differential_evolution

class BBOBMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.probability = 0.4

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        def crossover(individual1, individual2):
            if np.random.rand() < self.probability:
                new_individual = individual1 + individual2
            else:
                new_individual = individual1
            return new_individual

        def mutate(individual):
            if np.random.rand() < self.probability:
                new_individual = individual + np.random.uniform(-1, 1, self.dim)
            else:
                new_individual = individual
            return new_individual

        def evaluate_fitness(individual):
            return func(individual)

        def evaluateBBOB(func, population_size=100, budget=self.budget, dim=self.dim, bounds=self.bounds):
            population = np.random.uniform(bounds[0][0], bounds[0][1], (population_size, dim))
            for _ in range(budget):
                new_population = []
                for i in range(population_size):
                    individual1 = population[i]
                    individual2 = population[np.random.randint(0, population_size)]
                    new_individual = crossover(individual1, individual2)
                    new_individual = mutate(new_individual)
                    new_population.append(new_individual)
                population = np.array(new_population)
            fitness = np.array([evaluate_fitness(individual) for individual in population])
            return population, fitness

        population, fitness = evaluateBBOB(func)
        return np.argmin(fitness), np.min(fitness)

# Usage
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return x[0]**2 + 2*x[1]**2

def f3(x):
    return x[0]**3 + 2*x[1]**3

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

budget = 100
dim = 2
solution = BBOBMutation(budget, dim)
best_individual, best_fitness = solution(f1)
print(f"Best individual: {best_individual}, Best fitness: {best_fitness}")