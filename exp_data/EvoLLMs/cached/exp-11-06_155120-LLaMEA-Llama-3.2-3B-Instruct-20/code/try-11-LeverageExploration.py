import numpy as np
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class LeverageExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.x_best = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
        self.f_best = np.inf
        self.gp = None

    def __call__(self, func):
        for _ in range(self.budget):
            # Explore
            x_explore = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
            f_explore = func(x_explore)

            # Local search
            x_local = self.x_best
            f_local = func(x_local)

            # Leverage the best exploration point
            if f_explore < f_local:
                self.x_best = x_explore
                self.f_best = f_explore

            # Gaussian Process for informed exploration
            if self.gp is None:
                self.gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=0.1)
                self.gp.fit(self.x_best.reshape(-1, 1), self.f_best)

            # Predict the function value at the exploration point
            f_pred = self.gp.predict(x_explore.reshape(1, -1))[0]

            # Update the exploration point based on the predicted value
            if f_pred < f_explore:
                x_explore = self.gp.predict(x_explore.reshape(1, -1))[0]

            # Differential evolution for local search
            x_dev = differential_evolution(func, [(self.lower_bound, self.upper_bound) for _ in range(self.dim)])
            f_dev = func(x_dev.x)

            # Update the best point if the local search is better
            if f_dev < self.f_best:
                self.x_best = x_dev.x
                self.f_best = f_dev

# Example usage
def func(x):
    return x[0]**2 + x[1]**2

le = LeverageExploration(budget=100, dim=2)
le(func)
print("Best point:", le.x_best)
print("Best function value:", le.f_best)
