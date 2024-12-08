import numpy as np
import random

class GBestPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.w = 0.7298
        self.c1 = 1.49618
        self.c2 = 2.049912
        self.f = 0.5
        self.x = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fval = np.inf
        self.best_x = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the function at the current population
            fval = func(self.x)

            # Update the best solution
            if fval < self.fval:
                self.fval = fval
                self.best_x = self.x[np.argmin(fval)]
                self.x = np.array([self.best_x])

            # Refine the strategy with 0.35 probability
            refine = np.random.rand(self.population_size) < 0.35
            for i in range(self.population_size):
                if refine[i]:
                    # Update the individual using PSO and DE
                    v = self.w * np.random.uniform(0, 1, (self.dim,)) + self.c1 * np.abs(self.x[i] - self.best_x) + self.c2 * np.abs(self.x[i] - np.mean(self.x, axis=0)) ** self.f
                    self.x[i] = self.x[i] + v

                    # Limit the search space
                    self.x[i] = np.clip(self.x[i], self.lower_bound, self.upper_bound)

                    # Evaluate the function at the updated individual
                    fval = func(self.x[i])

                    # Update the best solution
                    if fval < self.fval:
                        self.fval = fval
                        self.best_x = self.x[np.argmin(fval)]

            # Select the best individual
            self.x = self.x[np.argmin(fval)]

        return self.fval, self.best_x

# Example usage:
def bbb_1d(x):
    return x[0]**2 + 0.3*np.sin(10*x[0]) + 0.3*np.sin(10*x[1])

def bbb_2d(x):
    return x[0]**2 + 0.3*np.sin(10*x[0]) + 0.3*np.sin(10*x[1]) + 0.3*np.sin(10*x[0])**2 + 0.3*np.sin(10*x[1])**2

def bbb_3d(x):
    return x[0]**2 + 0.3*np.sin(10*x[0]) + 0.3*np.sin(10*x[1]) + 0.3*np.sin(10*x[2]) + 0.3*np.sin(10*x[0])**2 + 0.3*np.sin(10*x[1])**2 + 0.3*np.sin(10*x[2])**2

def bbb_4d(x):
    return x[0]**2 + 0.3*np.sin(10*x[0]) + 0.3*np.sin(10*x[1]) + 0.3*np.sin(10*x[2]) + 0.3*np.sin(10*x[3]) + 0.3*np.sin(10*x[0])**2 + 0.3*np.sin(10*x[1])**2 + 0.3*np.sin(10*x[2])**2 + 0.3*np.sin(10*x[3])**2

def bbb_5d(x):
    return x[0]**2 + 0.3*np.sin(10*x[0]) + 0.3*np.sin(10*x[1]) + 0.3*np.sin(10*x[2]) + 0.3*np.sin(10*x[3]) + 0.3*np.sin(10*x[4]) + 0.3*np.sin(10*x[0])**2 + 0.3*np.sin(10*x[1])**2 + 0.3*np.sin(10*x[2])**2 + 0.3*np.sin(10*x[3])**2 + 0.3*np.sin(10*x[4])**2

# Define the BBOB test suite
BBOB_test_suite = [bbb_1d, bbb_2d, bbb_3d, bbb_4d, bbb_5d]

# Initialize the GBest-PSO-DE algorithm
budget = 100
dim = 10
algorithm = GBestPSODE(budget, dim)

# Evaluate the algorithm on the BBOB test suite
for func in BBOB_test_suite:
    fval, best_x = algorithm(func)
    print(f"Function: {func.__name__}, fval: {fval}, best_x: {best_x}")