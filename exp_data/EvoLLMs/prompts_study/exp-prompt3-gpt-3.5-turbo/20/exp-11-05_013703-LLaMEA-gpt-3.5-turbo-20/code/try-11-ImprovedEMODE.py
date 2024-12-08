import numpy as np

class ImprovedEMODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 10
        bounds = (-5.0, 5.0)
        
        def clip(x):
            return np.clip(x, bounds[0], bounds[1])
        
        def initialize_population():
            return np.random.uniform(bounds[0], bounds[1], size=(population_size, self.dim))
        
        def differential_evolution(population, cr, f):
            new_population = np.zeros_like(population)
            for i in range(population_size):
                target = population[i]
                candidates = np.random.choice(population, 6, replace=False)
                r1, r2, r3, r4, r5, r6 = candidates
                mutant = clip(r1 + f * (r2 - r3) + f * (r4 - r5) + f * (r6 - target))
                crossover_points = np.random.rand(self.dim) < cr
                offspring = np.where(crossover_points, mutant, target)
                new_population[i] = clip(offspring)
            return new_population
        
        population = initialize_population()
        evaluations = 0
        cr = 0.5  # initial crossover rate
        f = 0.5  # initial differential weight
        while evaluations < self.budget:
            offspring = differential_evolution(population, cr, f)
            for ind in offspring:
                fitness = func(ind)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            population = np.vstack((population, offspring))
            cr = max(0.1, cr - 0.05)  # decrease crossover rate
            f = min(1.0, f + 0.05)  # increase differential weight
        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution