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
        self.p = 0.3

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

                # Line search with probability refinement
                if random.random() < self.p:
                    # Refine the search direction
                    f_refined = func(x + self.p * v)
                    if f_refined < self.f_best:
                        self.x_best[_] = x
                        self.f_best = f_refined
                        self.pbest[_] = x
                        if f_refined < self.f:
                            self.f = f_refined
                            self.gbest = x
                else:
                    # Original search direction
                    f_original = func(x)
                    if f_original < self.f_best:
                        self.x_best[_] = x
                        self.f_best = f_original
                        self.pbest[_] = x
                        if f_original < self.f:
                            self.f = f_original
                            self.gbest = x

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