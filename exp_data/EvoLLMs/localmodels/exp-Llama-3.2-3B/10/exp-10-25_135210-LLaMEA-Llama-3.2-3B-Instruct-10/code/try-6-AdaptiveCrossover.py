import numpy as np

class AdaptiveCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x_best = np.random.uniform(self.bounds[0][0], self.bounds[0][1], dim)
        self.f_best = np.inf
        self.crossover_rate = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            x = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
            f = func(x)

            if f < self.f_best:
                self.x_best = x
                self.f_best = f

            # Adapt crossover rate based on the problem's difficulty
            if np.random.rand() < self.crossover_rate:
                # Crossover
                if np.random.rand() < 0.5:
                    x1, x2 = np.random.split(x, 2)
                    x1 = np.clip(x1, self.bounds[0][0], self.bounds[0][1])
                    x2 = np.clip(x2, self.bounds[0][0], self.bounds[0][1])
                    x = (x1 + x2) / 2
                else:
                    x1, x2 = np.random.split(x, 2)
                    x1 = np.clip(x1, self.bounds[0][0], self.bounds[0][1])
                    x2 = np.clip(x2, self.bounds[0][0], self.bounds[0][1])
                    x = x1 + x2

            # Update x_best
            x_best_new = x + np.random.uniform(-1, 1, self.dim)
            x_best_new = np.clip(x_best_new, self.bounds[0][0], self.bounds[0][1])
            f_best_new = func(x_best_new)
            if f_best_new < self.f_best:
                self.x_best = x_best_new
                self.f_best = f_best_new

# Test the algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
alg = AdaptiveCrossover(budget, dim)
alg()