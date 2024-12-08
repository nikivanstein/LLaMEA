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
        self.population_size = 10

    def __call__(self, func):
        for i in range(self.budget):
            # Generate random initial population
            x = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

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

            # Update population with new best individual
            x = np.vstack((x, self.x_best))

            # Evaluate new population
            f = func(x)

            # Store best individual
            self.x_best = x[:, [np.argmin(f, axis=0)]]
            self.f_best = np.min(f, axis=0)

            # Refine strategy with probability 0.45
            if np.random.rand() < 0.45:
                for individual in x:
                    # Randomly select two individuals to refine
                    i1, i2 = np.random.randint(0, self.population_size, 2)

                    # Calculate new individual using GDE
                    new_individual = self.gde_refine(individual, x[i1], x[i2])

                    # Replace individual with new individual
                    x[i1] = new_individual

    def gde_refine(self, individual, parent1, parent2):
        # Calculate difference vector
        diff = parent1 - individual

        # Calculate new individual
        new_individual = individual + self.p * diff

        # Calculate new function value
        f_new = self.func(new_individual)

        # Return new individual
        return new_individual

def func(x):
    return np.sum(x**2)

gde = GDE(budget=10, dim=2)
gde(func)