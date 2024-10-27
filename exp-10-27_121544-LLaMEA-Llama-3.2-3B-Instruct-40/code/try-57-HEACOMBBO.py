import numpy as np
import random
import copy
from scipy.optimize import differential_evolution

class HEACOMBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.fitness = np.zeros(self.budget)
        self.x = np.zeros((self.budget, self.dim))
        self.logger = []

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        for i in range(self.budget):
            if i > 0:
                new_individual = copy.deepcopy(self.x[i-1])
                if random.random() < 0.4:
                    new_individual = self.mutate(new_individual)
                self.fitness[i] = func(new_individual)
                self.x[i] = new_individual

            if i < self.budget // 2:
                new_individual = copy.deepcopy(self.x[i])
                new_individual = self.crossover(new_individual)
                self.fitness[i] = func(new_individual)
                self.x[i] = new_individual

        best_individual = self.x[np.argmin(self.fitness)]
        return np.min(self.fitness), np.min(self.fitness)

    def mutate(self, individual):
        mutated_individual = list(individual)
        for i in range(self.dim):
            if random.random() < 0.1:
                mutated_individual[i] += np.random.uniform(-1, 1)
                if mutated_individual[i] < self.bounds[i][0]:
                    mutated_individual[i] = self.bounds[i][0]
                elif mutated_individual[i] > self.bounds[i][1]:
                    mutated_individual[i] = self.bounds[i][1]
        return mutated_individual

    def crossover(self, individual):
        crossover_point = random.randint(1, self.dim - 1)
        child1 = list(individual[:crossover_point])
        child2 = list(individual[crossover_point:])
        for i in range(self.dim):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return child1 + child2

# Usage
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

if __name__ == "__main__":
    heacombbo = HEACOMBBO(100, 2)
    print(heacombbo(heacombbo.f1))