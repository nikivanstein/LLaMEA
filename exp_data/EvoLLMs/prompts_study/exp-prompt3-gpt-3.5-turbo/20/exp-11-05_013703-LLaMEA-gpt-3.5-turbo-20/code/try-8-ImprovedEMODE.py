import numpy as np

class ImprovedEMODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.bounds = (-5.0, 5.0)
        self.cr = 0.9
        self.f = 0.8

    def __call__(self, func):
        
        def clip(x):
            return np.clip(x, self.bounds[0], self.bounds[1])
        
        def initialize_population():
            return np.random.uniform(self.bounds[0], self.bounds[1], size=(self.population_size, self.dim))
                
        population = initialize_population()
        evaluations = 0
        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            for i in range(self.population_size):
                target = population[i]
                r1, r2, r3 = np.random.choice(population, 3, replace=False)
                mutation_factor = np.random.normal(self.f, 0.1)
                crossover_rate = np.random.normal(self.cr, 0.1)
                
                mutant = clip(r1 + mutation_factor * (r2 - r3))
                crossover_points = np.random.rand(self.dim) < crossover_rate
                offspring = np.where(crossover_points, mutant, target)
                
                new_population[i] = clip(offspring)
                
                fitness = func(offspring)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            
            population = new_population
        
        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution