import numpy as np

class GDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x_best = np.zeros((budget, dim))
        self.f_best = np.inf
        self.x_new = np.zeros((budget, dim))
        self.f_new = np.inf
        self.p = 0.5
        self.q = 0.5
        self.r = 0.45

    def __call__(self, func):
        for i in range(self.budget):
            # Generate random initial population
            x = np.random.uniform(-5.0, 5.0, (self.dim, 1))

            # Evaluate initial population
            f = func(x)

            # Store best individual
            self.x_best[i] = x[0]
            self.f_best = f

            # Differential evolution
            for j in range(1, self.dim):
                for k in range(j, self.dim):
                    # Calculate difference vector
                    diff = x[k] - x[j]

                    # Calculate new individual
                    x_new = x[j] + self.q * diff

                    # Calculate new function value
                    f_new = func(x_new)

                    # Update best individual if new function value is better
                    if f_new < self.f_best:
                        self.x_best[i] = x_new
                        self.f_best = f_new

            # Refine the best individual
            if self.f_best!= np.inf:
                for _ in range(int(self.r * self.budget)):
                    idx = np.random.choice(self.dim, 2, replace=False)
                    idx.sort()
                    x_new = self.x_best[i] + self.p * (self.x_best[i][idx[1]] - self.x_best[i][idx[0]])
                    f_new = func(x_new)
                    if f_new < self.f_best:
                        self.x_best[i] = x_new
                        self.f_best = f_new

            # Update population with new best individual
            self.x_best = np.vstack((self.x_best, self.x_new))

            # Evaluate new population
            f = func(self.x_best)

            # Store best individual
            self.x_best = self.x_best[:, [np.argmin(f, axis=0)]]
            self.f_best = np.min(f, axis=0)

# Example usage:
def func(x):
    return np.sum(x**2)

gde = GDE(budget=10, dim=2)
gde(func)