import numpy as np
import scipy.optimize as optimize

class ProbabilisticAdaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x_best = np.random.uniform(self.bounds[0][0], self.bounds[0][1], dim)
        self.f_best = np.inf
        self.p_adapt = 0.1
        self.algorithms = [np.random.uniform(self.bounds[0][0], self.bounds[0][1], dim) for _ in range(10)]

    def __call__(self, func):
        for i in range(self.budget):
            # Select algorithm to use
            algorithm_idx = np.random.choice(len(self.algorithms), p=np.ones(len(self.algorithms))/len(self.algorithms))
            x = self.algorithms[algorithm_idx]

            # Evaluate function
            f = func(x)

            # Update best solution
            if f < self.f_best:
                self.x_best = x
                self.f_best = f

            # Adapt algorithm
            if np.random.rand() < self.p_adapt:
                # Change individual line
                individual_idx = np.random.choice(len(self.algorithms))
                individual = self.algorithms[individual_idx]
                individual[np.random.randint(0, self.dim)] += np.random.uniform(-1, 1)
                individual = np.clip(individual, self.bounds[0][0], self.bounds[0][1])
                self.algorithms[individual_idx] = individual

# Test the algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
alg = ProbabilisticAdaptive(budget, dim)
alg()