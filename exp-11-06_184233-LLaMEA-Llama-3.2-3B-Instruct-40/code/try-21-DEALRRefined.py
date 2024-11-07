import numpy as np
import random

class DEALRRefined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=(budget, dim))
        self.f = np.zeros(budget)
        self.pop = np.zeros((budget, dim))
        self.LR = 0.5 + 0.1 * np.random.uniform(-0.1, 0.1, size=1)
        self.GR = 0.5 + 0.1 * np.random.uniform(-0.1, 0.1, size=1)
        self.exploitation = 0.7
        self.exploration = 0.3
        self.crossover_prob = 0.5
        self.pop_size = budget // 2
        self.pop_init = np.random.uniform(-5.0, 5.0, size=(self.pop_size, dim))
        self.f_init = np.random.uniform(-5.0, 5.0, size=(self.pop_size,))

    def __call__(self, func):
        for i in range(self.budget):
            if i > 0:
                self.pop[i] = self.f_init + self.exploitation * np.random.uniform(-1, 1, size=self.dim) + (1-self.exploration) * np.random.uniform(-1, 1, size=self.dim)
            f_i = func(self.pop[i])
            if f_i < self.f[i]:
                self.x[i] = self.pop[i]
                self.f[i] = f_i
                if np.random.rand() < self.crossover_prob:
                    self.x[i] = self.recombine(self.x[i], self.x[i-1])
            if self.f[i] < self.f[0]:
                self.f[0] = self.f[i]
                self.pop[0] = self.pop[i]
        return self.x[np.argmin(self.f)], np.min(self.f)

    def recombine(self, parent1, parent2):
        alpha = np.random.uniform(0.5, 1.0)
        child = alpha * parent1 + (1-alpha) * parent2
        return child

    def adjust_pop_size(self, f_values):
        if np.min(f_values) < np.mean(f_values):
            self.pop_size = int(self.pop_size * 1.1)
            self.pop_init = np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
            self.f_init = np.random.uniform(-5.0, 5.0, size=(self.pop_size,))
        else:
            self.pop_size = int(self.pop_size * 0.9)
            self.pop_init = np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
            self.f_init = np.random.uniform(-5.0, 5.0, size=(self.pop_size,))

# Example usage:
def func(x):
    return np.sum(x**2)

dealr_refined = DEALRRefined(budget=100, dim=10)
x, f = dealr_refined(func)
print(f'Optimal solution: x = {x}, f = {f}')