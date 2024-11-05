import numpy as np

class ImprovedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.bounds = (-5.0, 5.0)
        self.adaptive_cr = 0.5
        self.adaptive_f = 0.5
        self.cr_range = (0.1, 1.0)
        self.f_range = (0.1, 0.9)

    def __call__(self, func):
        
        def clip(x):
            return np.clip(x, self.bounds[0], self.bounds[1])
        
        def initialize_population():
            return np.random.uniform(self.bounds[0], self.bounds[1], size=(self.population_size, self.dim))
        
        def differential_evolution(population, cr, f):
            new_population = np.zeros_like(population)
            for i in range(self.population_size):
                target = population[i]
                r1, r2, r3 = np.random.choice(population, 3, replace=False)
                mutant = clip(r1 + f * (r2 - r3))
                crossover_points = np.random.rand(self.dim) < cr
                offspring = np.where(crossover_points, mutant, target)
                new_population[i] = clip(offspring)
            return new_population
        
        population = initialize_population()
        evaluations = 0
        while evaluations < self.budget:
            cr = np.random.uniform(*self.cr_range)
            f = np.random.uniform(*self.f_range)
            offspring = differential_evolution(population, cr, f)
            for ind in offspring:
                fitness = func(ind)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            population = np.vstack((population, offspring))
        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution