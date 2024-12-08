# import numpy as np

class FireflyLevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.fireflies = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.best = np.zeros(self.dim)
        self.best_score = np.inf

    def levy_flight(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.math.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        s = np.random.normal(0, sigma, self.dim)
        u = np.random.normal(0, 1, self.dim)
        v = u / abs(u) ** (1 / beta)
        levy = 0.01 * s * v
        return levy

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.pop_size):
                intensity_i = 1 / (1 + np.sum((self.fireflies[i] - self.best) ** 2))
                for j in range(self.pop_size):
                    if func(self.fireflies[j]) < func(self.fireflies[i]):
                        beta = 0.2 * np.exp(-0.2 * _)
                        self.fireflies[i] += intensity_i * (self.fireflies[j] - self.fireflies[i]) + beta * self.levy_flight()
                if func(self.fireflies[i]) < func(self.best):
                    self.best = self.fireflies[i]
            beta = 0.5 + 0.2 * np.random.rand()
            self.fireflies += beta * np.random.uniform(-1, 1, (self.pop_size, self.dim))
            # Dynamic Mutation Rates
            F = 0.5 + 0.5 * np.random.rand()
            CR = 0.1 + 0.9 * np.random.rand()
            r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
            mutant = self.fireflies[r1] + F * (self.fireflies[r2] - self.fireflies[r3])
            crossover = np.random.rand(self.dim) < CR
            trial = np.where(crossover, mutant, self.fireflies[i])
            trial_fitness = func(trial)
            if trial_fitness < func(self.fireflies[i]):
                self.fireflies[i] = trial
        return self.best