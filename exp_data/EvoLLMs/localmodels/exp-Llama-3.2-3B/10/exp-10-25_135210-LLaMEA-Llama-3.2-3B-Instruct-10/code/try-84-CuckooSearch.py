import numpy as np
import random

class CuckooSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x_best = np.random.uniform(self.bounds[0][0], self.bounds[0][1], dim)
        self.f_best = np.inf
        self.p = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            # Generate a new solution
            x = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
            f = func(x)

            # Check if the new solution is better than the best solution found so far
            if f < self.f_best:
                self.x_best = x
                self.f_best = f

            # If the new solution is better, mutate it with a probability of 0.1
            if random.random() < self.p:
                # Generate a random mutation
                mutation = np.random.uniform(-1.0, 1.0, self.dim)
                x += mutation

                # Check if the mutated solution is within the bounds
                x = np.clip(x, self.bounds[0][0], self.bounds[0][1])

                # Evaluate the mutated solution
                f = func(x)

                # Check if the mutated solution is better than the best solution found so far
                if f < self.f_best:
                    self.x_best = x
                    self.f_best = f

# Test the algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
alg = CuckooSearch(budget, dim)
alg(func)