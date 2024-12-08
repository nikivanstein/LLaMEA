import numpy as np
import random

class CMA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=(budget, dim))
        self.f = np.zeros(budget)
        self.x_best = self.x.copy()
        self.f_best = np.inf

    def __call__(self, func):
        for i in range(self.budget):
            if i > 0:
                self.x[i] = self.x_best + np.random.normal(0, 0.1, size=(self.dim,))
                self.f[i] = func(self.x[i])

            if self.f[i] < self.f_best:
                self.x_best = self.x[i].copy()
                self.f_best = self.f[i]

        return self.x_best, self.f_best

def refine_solution(x_best, f_best, x, f):
    for i in range(len(x)):
        if random.random() < 0.3:
            x[i] += np.random.uniform(-0.1, 0.1)
            f[i] = func(x[i])
            if f[i] < f_best:
                x_best = x[i].copy()
                f_best = f[i]

    return x_best, f_best

def evaluate_function(func, x, f):
    if func(x) < f:
        f = func(x)
    return f

def main():
    func = lambda x: np.sum(x**2)  # example function
    cma = CMA(budget=100, dim=5)
    x_best, f_best = cma(func)
    print("Initial Solution: ", x_best, f_best)

    for _ in range(10):
        x_best, f_best = refine_solution(x_best, f_best, cma.x, cma.f)
        x_best, f_best = evaluate_function(func, x_best, f_best)
        print("Refined Solution: ", x_best, f_best)

if __name__ == "__main__":
    main()