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
        def clip_to_bounds(x):
            return np.clip(x, self.lower_bound, self.upper_bound)
        
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness_values = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget - self.population_size):
            mutants = population[np.random.choice(range(self.population_size), (self.population_size, 3), replace=True)]
            crossover = np.random.rand(self.population_size, self.dim) < self.cr
            
            new_population = np.where(crossover, clip_to_bounds(population + self.f * (mutants[:, 0] - mutants[:, 1])), population)
            new_population = np.where(~crossover, population, new_population)
            
            new_fitness_values = np.array([func(ind) for ind in new_population])
            improved_indices = new_fitness_values < fitness_values
            population[improved_indices] = new_population[improved_indices]
            fitness_values[improved_indices] = new_fitness_values[improved_indices]
        
        best_index = np.argmin(fitness_values)
        return population[best_index]