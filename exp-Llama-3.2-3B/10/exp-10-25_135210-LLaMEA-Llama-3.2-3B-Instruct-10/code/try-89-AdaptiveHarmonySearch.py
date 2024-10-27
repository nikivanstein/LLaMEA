import numpy as np
import random

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.memory = np.zeros((budget, self.dim))
        self.fitness = np.zeros(budget)
        self.x_best = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.f_best = np.inf
        self.p = 0.1
        self.m = 0.5

    def __call__(self, func):
        for _ in range(self.budget):
            x = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
            f = func(x)

            if f < self.f_best:
                self.x_best = x
                self.f_best = f

            # Update memory
            self.memory[_] = x
            self.fitness[_] = f

            # Probability-based mutation
            if random.random() < self.p:
                i = random.randint(0, _)
                x = self.memory[i] + np.random.uniform(-1, 1, self.dim)
                f = func(x)

                if f < self.f_best:
                    self.x_best = x
                    self.f_best = f

            # Adaptive harmony search
            if _ > 1:
                for i in range(_):
                    if random.random() < self.m:
                        x = self.memory[i] + np.random.uniform(-1, 1, self.dim)
                        f = func(x)

                        if f < self.f_best:
                            self.x_best = x
                            self.f_best = f

# Test the algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
alg = AdaptiveHarmonySearch(budget, dim)
alg()