import numpy as np
import random

class SwarmPSODEProb:
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
        self.prob = 0.3

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

            # Update particles with probability
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
                if random.random() < self.prob and f < self.f_best:
                    self.x_best[_] = x
                    self.f_best = f
                    self.pbest[_] = x
                    if f < self.f:
                        self.f = f
                        self.gbest = x

# Example usage:
funcs = [lambda x: x**2, lambda x: x**4, lambda x: x**6, lambda x: x**8, lambda x: x**10, lambda x: x**12, lambda x: x**14, lambda x: x**16, lambda x: x**18, lambda x: x**20, lambda x: x**22, lambda x: x**24, lambda x: x**26, lambda x: x**28, lambda x: x**30, lambda x: x**32, lambda x: x**34, lambda x: x**36, lambda x: x**38, lambda x: x**40, lambda x: x**42, lambda x: x**44, lambda x: x**46, lambda x: x**48]
swarmPSODEProb = SwarmPSODEProb(100, 5)
for func in funcs:
    swarmPSODEProb(func)
    print(f"Function: {func.__name__}")
    print(f"Best fitness: {swarmPSODEProb.f_best}")
    print(f"Best x: {swarmPSODEProb.x_best[np.argmax(swarmPSODEProb.f_best)]}")
    print(f"")