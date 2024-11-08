import numpy as np

class ImprovedDEAdaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.f = 0.5
        self.cr = 0.9
        self.population_size = 10
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness_values = np.array([func(ind) for ind in population])
        
        mutants_idx = np.random.choice(self.population_size, (self.budget - self.population_size, 2), replace=True)
        crossover = np.random.rand(self.budget - self.population_size, self.dim) < self.cr
        mutants = population[mutants_idx]

        for i in range(self.budget - self.population_size):
            new_population = population + self.f * (mutants[i, 0] - mutants[i, 1])
            new_population = np.where(crossover[i], np.clip(new_population, self.lower_bound, self.upper_bound), population)

            new_fitness = func(new_population)
            if new_fitness < fitness_values[0]:
                population[0] = new_population
                fitness_values[0] = new_fitness
        
        return population[np.argmin(fitness_values)]