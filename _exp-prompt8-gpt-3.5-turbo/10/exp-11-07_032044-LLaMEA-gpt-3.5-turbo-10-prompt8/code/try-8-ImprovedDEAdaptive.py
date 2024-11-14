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
            for i in range(self.population_size):
                a, b, c = population[np.random.choice([idx for idx in range(self.population_size) if idx != i], 3, replace=False)]
                mutant = clip_to_bounds(a + self.f * (b - c))
                crossover = np.random.rand(self.dim) < self.cr
                population[i] = np.where(crossover, mutant, population[i])
                
                new_fitness = func(population[i])
                if new_fitness < fitness_values[i]:
                    fitness_values[i] = new_fitness
        
        best_index = np.argmin(fitness_values)
        return population[best_index]