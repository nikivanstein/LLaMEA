import numpy as np

class ImprovedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.bounds = (-5.0, 5.0)
        self.adaptive_cr = 0.9
        self.adaptive_f = 0.8

    def __call__(self, func):
        
        def clip(x):
            return np.clip(x, self.bounds[0], self.bounds[1])
        
        def initialize_population():
            return np.random.uniform(self.bounds[0], self.bounds[1], size=(self.population_size, self.dim))
        
        def differential_evolution(population):
            new_population = np.zeros_like(population)
            for i in range(self.population_size):
                target = population[i]
                r1, r2, r3 = np.random.choice(population, 3, replace=False)
                adaptive_f = np.random.normal(self.adaptive_f, 0.1)
                adaptive_cr = np.random.normal(self.adaptive_cr, 0.1)
                mutant = clip(r1 + adaptive_f * (r2 - r3))
                crossover_points = np.random.rand(self.dim) < adaptive_cr
                offspring = np.where(crossover_points, mutant, target)
                new_population[i] = clip(offspring)
            return new_population
        
        population = initialize_population()
        evaluations = 0
        while evaluations < self.budget:
            offspring = differential_evolution(population)
            for ind in offspring:
                fitness = func(ind)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            population = np.vstack((population, offspring))
        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution