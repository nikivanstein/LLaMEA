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
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def __call__(self, func):
        for i in range(self.budget):
            # Evaluate initial population
            f = func(self.population)

            # Store best individual
            self.x_best[i] = self.population[np.argmin(f)]
            self.f_best = np.min(f)

            # Differential evolution
            for j in range(self.dim):
                for k in range(j+1, self.dim):
                    # Calculate difference vector
                    diff = self.population[k] - self.population[j]

                    # Calculate new individual
                    x_new = self.population[j] + self.q * diff

                    # Calculate new function value
                    f_new = func(x_new)

                    # Update best individual if new function value is better
                    if f_new < self.f_best:
                        self.x_best[i] = x_new
                        self.f_best = f_new

            # Update population with new best individual
            self.population = np.vstack((self.population, self.x_best[i]))

# Example usage:
def func(x):
    return np.sum(x**2)

gde = GDE(budget=10, dim=2)
gde(func)