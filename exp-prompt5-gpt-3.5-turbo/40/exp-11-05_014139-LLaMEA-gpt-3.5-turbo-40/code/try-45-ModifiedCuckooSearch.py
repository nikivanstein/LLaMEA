import numpy as np

class ModifiedCuckooSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.pa = 0.25  # Discovery rate
        self.alpha = 0.1  # Step size scaling factor
        self.sigma = 0.1
        self.sigma_min = 0.01
        self.sigma_max = 0.2

    def levy_flight(self):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (abs(v) ** (1 / beta))
        return step

    def __call__(self, func):
        nest = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in nest])

        for _ in range(self.budget):
            new_nest = np.copy(nest)

            # Generate new solutions
            step_size = self.alpha * self.levy_flight()
            new_nest += step_size

            # Evaluate new solutions
            new_fitness = np.array([func(ind) for ind in new_nest])

            # Replace the old solutions with new ones if better
            replace_idxs = new_fitness < fitness
            nest[replace_idxs] = new_nest[replace_idxs]
            fitness[replace_idxs] = new_fitness[replace_idxs]

            # Abandon a fraction of worst solutions and replace them with new random solutions
            replace_nests = np.random.rand(self.population_size, self.dim) < self.pa
            new_nests = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
            nest[replace_nests] = new_nests[replace_nests]
            fitness[replace_nests] = np.array([func(ind) for ind in new_nests])

        return nest[np.argmin(fitness)]