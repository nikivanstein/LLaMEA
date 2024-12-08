import numpy as np
from scipy.optimize import differential_evolution
import time
import random

class ProbabilisticDDECM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], size=dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.min(func(self.x0))
        
        t0 = time.time()
        res = differential_evolution(func, self.bounds, x0=self.x0, maxiter=self.budget)
        t1 = time.time()
        
        # Calculate the diversity metric
        diversity = np.mean([np.sum(np.abs(func(x) - func(res.x))) for x in np.random.uniform(self.bounds[0][0], self.bounds[0][1], size=(self.dim, self.budget))])
        
        # Apply probabilistic crossover to increase diversity
        mutation_rate = 0.1
        mutation_x = np.zeros((self.dim, self.budget))
        for i in range(self.dim):
            for j in range(self.budget):
                if random.random() < mutation_rate:
                    mutation_x[i, j] = np.random.uniform(self.bounds[i][0], self.bounds[i][1])
                else:
                    mutation_x[i, j] = res.x[i]
        res.x = np.concatenate((res.x[:self.dim], mutation_x), axis=0)
        
        # Apply diversity-driven mutation to increase diversity
        diversity_driven_mutation_rate = 0.05
        for i in range(self.dim):
            for j in range(self.budget):
                if random.random() < diversity_driven_mutation_rate:
                    mutation_x[i, j] = np.random.uniform(self.bounds[i][0], self.bounds[i][1])
                    res.x[i] = mutation_x[i, j]
        
        # Return the optimized function value and the diversity metric
        return res.fun, diversity

# Example usage:
if __name__ == "__main__":
    from blackbox_optimization import bbbob
    from numpy import testing

    for i in range(24):
        def func(x):
            return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2

        pddecm = ProbabilisticDDECM(budget=50, dim=10)
        best_x, best_f = pddecm(func)
        print(f"Function {i+1}: x = {best_x}, f(x) = {best_f}")

        # Test the optimization algorithm
        x = np.linspace(-5.0, 5.0, 100)
        y = func(x)
        testing.assert_allclose(np.min(y), pddecm(func)(func, 0)[1], atol=1e-3)