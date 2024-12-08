import numpy as np

class EnhancedEvolutionaryStrategies:
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
            self.mutate_population(func)
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

    def mutate_population(self, func):
        for i, ind in enumerate(self.population):
            indices = [idx for idx in range(len(self.population)) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = self.population[a] + 0.5 * (self.population[b] - self.population[c])
            new_ind = ind + self.sigma * mutant
            if func(new_ind) < func(ind):
                self.population[i] = new_ind