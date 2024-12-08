import numpy as np

class EMODE_AdaptiveMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 10
        cr = 0.9
        bounds = (-5.0, 5.0)
        
        def clip(x):
            return np.clip(x, bounds[0], bounds[1])
        
        def initialize_population():
            return np.random.uniform(bounds[0], bounds[1], size=(population_size, self.dim))
        
        def differential_evolution(population, f):
            new_population = np.zeros_like(population)
            for i in range(population_size):
                target = population[i]
                r1, r2, r3 = np.random.choice(population, 3, replace=False)
                f_mutant = np.clip(np.abs(np.random.normal(f, 0.1)))
                mutant = clip(r1 + f_mutant * (r2 - r3))
                crossover_points = np.random.rand(self.dim) < cr
                offspring = np.where(crossover_points, mutant, target)
                new_population[i] = clip(offspring)
            return new_population
        
        population = initialize_population()
        evaluations = 0
        f = 0.8
        while evaluations < self.budget:
            offspring = differential_evolution(population, f)
            for ind in offspring:
                fitness = func(ind)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            population = np.vstack((population, offspring))
        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution