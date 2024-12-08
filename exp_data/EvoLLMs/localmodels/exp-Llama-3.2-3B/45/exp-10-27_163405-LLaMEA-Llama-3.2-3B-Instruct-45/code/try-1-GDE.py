import numpy as np

class GDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x_best = np.zeros((budget, dim))
        self.f_best = np.inf
        self.x_new = np.zeros((budget, dim))
        self.f_new = np.inf
        self.p = 0.45  # probability to change individual
        self.q = 0.5
        self.population = np.zeros((budget, dim))

    def __call__(self, func):
        for i in range(self.budget):
            # Generate random initial population
            self.population[i] = np.random.uniform(-5.0, 5.0, (self.dim, 1))

            # Evaluate initial population
            f = func(self.population[i])

            # Store best individual
            self.x_best[i] = self.population[i][0]
            self.f_best = f

            # Differential evolution
            for j in range(1, self.dim):
                for k in range(j, self.dim):
                    # Calculate difference vector
                    diff = self.population[k] - self.population[j]

                    # Calculate new individual
                    new_individual = self.population[j] + self.q * diff

                    # Calculate new function value
                    new_f = func(new_individual)

                    # Update best individual if new function value is better
                    if new_f < self.f_best:
                        self.x_best[i] = new_individual
                        self.f_best = new_f

            # Update population with new best individual
            self.population[i] = self.x_best[i]

            # Apply mutation
            for j in range(self.dim):
                if np.random.rand() < self.p:
                    self.population[i, j] += np.random.uniform(-1.0, 1.0)

# Example usage:
def func(x):
    return np.sum(x**2)

gde = GDE(budget=10, dim=2)
gde(func)