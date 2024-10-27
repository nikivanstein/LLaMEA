import numpy as np
import random

class SwarmPSODERandom:
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
        self.random_prob = 0.3

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

                # Random mutation
                if random.random() < self.random_prob:
                    # Randomly choose a dimension
                    idx = np.random.randint(0, self.dim)
                    # Randomly change the value of the chosen dimension
                    x[idx] += np.random.uniform(-1.0, 1.0)
                    x = np.clip(x, -5.0, 5.0)  # Clip to bounds
                    f = func(x)
                    if f < self.f_best:
                        self.x_best[_] = x
                        self.f_best = f
                        self.pbest[_] = x
                        if f < self.f:
                            self.f = f
                            self.gbest = x