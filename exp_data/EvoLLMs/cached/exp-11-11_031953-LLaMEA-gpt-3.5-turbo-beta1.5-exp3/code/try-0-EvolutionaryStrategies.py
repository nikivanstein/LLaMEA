import numpy as np

class EvolutionaryStrategies:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.sigma = 0.1
        self.best_solution = None
        self.best_fitness = np.inf

    def __call__(self, func):
        self.initialize_population()
        for _ in range(self.budget // self.population_size):
            self.mutate_population()
            self.evaluate_population(func)
        return self.best_solution

    def initialize_population(self):
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

    def evaluate_population(self, func):
        for ind in self.population:
            fitness = func(ind)
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = np.copy(ind)

    def mutate_population(self):
        for ind in self.population:
            mutation = np.random.normal(0, self.sigma, size=self.dim)
            ind += mutation
            ind = np.clip(ind, -5.0, 5.0)