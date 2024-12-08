import numpy as np

class NHDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.cr = 0.9  # Crossover rate
        self.f = 0.5   # Differential weight
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

        population = initialize_population()
        best_solution = population[np.argmin([func(individual) for individual in population])

        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.f * (b - c), self.lb, self.ub)
                crossover = np.random.rand(self.dim) < self.cr
                new_solution = np.where(crossover, mutant, population[i])

                if func(new_solution) < func(population[i]):
                    population[i] = new_solution

                    if func(new_solution) < func(best_solution):
                        best_solution = new_solution

        return best_solution