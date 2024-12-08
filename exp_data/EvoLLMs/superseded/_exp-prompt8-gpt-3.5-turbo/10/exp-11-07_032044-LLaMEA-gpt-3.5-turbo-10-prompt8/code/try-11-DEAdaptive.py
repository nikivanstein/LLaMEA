import numpy as np

class DEAdaptive:
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
        
        def initialize_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        
        population = initialize_population()
        fitness_values = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget - self.population_size):
            new_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = clip_to_bounds(a + self.f * (b - c))
                crossover = np.random.rand(self.dim) < self.cr
                new_population[i] = np.where(crossover, mutant, population[i])
            
            new_fitness_values = np.array([func(ind) for ind in new_population])
            improved_indices = new_fitness_values < fitness_values
            population[improved_indices] = new_population[improved_indices]
            fitness_values[improved_indices] = new_fitness_values[improved_indices]
        
        best_index = np.argmin(fitness_values)
        return population[best_index]