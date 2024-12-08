import numpy as np

class AdaptiveCuckooSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 * dim
        self.pa = 0.25
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _get_fitness(self, population, func):
        return np.array([func(individual) for individual in population])

    def _levy_flight(self, step_size):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, step_size)
        v = np.random.normal(0, 1, step_size)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def _update_nests(self, nests, new_nests, func):
        fitness = self._get_fitness(nests, func)
        new_fitness = self._get_fitness(new_nests, func)
        for i in range(len(nests)):
            if new_fitness[i] < fitness[i]:
                nests[i] = new_nests[i]
        return nests

    def _random_walk(self, nest, step_size):
        new_nest = nest + self._levy_flight(step_size)
        return np.clip(new_nest, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        nests = self._initialize_population()
        step_size = 0.1 * (self.upper_bound - self.lower_bound)

        for _ in range(self.budget):
            new_nests = np.array([self._random_walk(nest, step_size) if np.random.rand() < self.pa else nest for nest in nests])
            nests = self._update_nests(nests, new_nests, func)

        best_solution = nests[np.argmin(self._get_fitness(nests, func))]
        return best_solution