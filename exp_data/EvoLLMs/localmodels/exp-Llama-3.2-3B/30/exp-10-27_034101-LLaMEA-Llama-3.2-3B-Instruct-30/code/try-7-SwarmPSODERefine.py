import numpy as np
import random

class SwarmPSODERefine:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x_best = np.zeros((budget, dim))
        self.f_best = np.inf
        self.pbest = np.zeros((budget, dim))
        self.gbest = np.zeros(dim)
        self.c1 = 2.0
        self.c2 = 2.0
        self.f = 0.5
        self.refine_prob = 0.3

    def __call__(self, func):
        for _ in range(self.budget):
            # Initialize particles
            x = np.random.uniform(-5.0, 5.0, (self.dim,))
            self.x_best[_] = x
            self.f_best = func(x)
            self.pbest[_] = x
            self.gbest = x

            # Update gbest
            if self.f_best < self.f:
                self.f = self.f_best
                self.gbest = self.x_best[_]

            # Update particles
            for _ in range(100):
                r1 = random.random()
                r2 = random.random()
                v = np.zeros(self.dim)
                v += self.c1 * r1 * (self.gbest - self.x_best[_])
                v += self.c2 * r2 * (self.pbest[_] - self.x_best[_])
                v = v * self.f
                x += v
                x = np.clip(x, -5.0, 5.0)  # Clip to bounds
                f = func(x)
                if f < self.f_best:
                    self.x_best[_] = x
                    self.f_best = f
                    self.pbest[_] = x
                    if f < self.f:
                        self.f = f
                        self.gbest = x

            # Refine strategy with 30% probability
            if random.random() < self.refine_prob:
                # Randomly select two particles to refine
                i1, i2 = random.sample(range(self.budget), 2)
                # Calculate new particle using PSO and DE
                v = np.zeros(self.dim)
                v += self.c1 * (self.gbest - self.x_best[i1])
                v += self.c2 * (self.pbest[i2] - self.x_best[i2])
                v = v * self.f
                x_refine = self.x_best[i1] + v
                f_refine = func(x_refine)
                # Update best particle if new particle is better
                if f_refine < self.f_best:
                    self.x_best[i2] = x_refine
                    self.f_best = f_refine
                    self.pbest[i2] = x_refine
                    if f_refine < self.f:
                        self.f = f_refine
                        self.gbest = x_refine

def evaluate(func, x):
    return func(x)

def bbb_test_suite(funcs):
    results = {}
    for func in funcs:
        results[func.__name__] = []
        for _ in range(10):
            x = np.random.uniform(-5.0, 5.0, (100,))
            f = func(x)
            results[func.__name__].append(f)
    return results

# Example usage:
funcs = [evaluate, evaluate, evaluate, evaluate, evaluate, evaluate, evaluate, evaluate, evaluate, evaluate, evaluate, evaluate, evaluate, evaluate, evaluate, evaluate, evaluate, evaluate, evaluate, evaluate, evaluate, evaluate]
results = bbb_test_suite(funcs)
for func_name, results in results.items():
    print(f"Function: {func_name}")
    print(f"Mean: {np.mean(results)}")
    print(f"Median: {np.median(results)}")
    print(f"Standard Deviation: {np.std(results)}")
    print()