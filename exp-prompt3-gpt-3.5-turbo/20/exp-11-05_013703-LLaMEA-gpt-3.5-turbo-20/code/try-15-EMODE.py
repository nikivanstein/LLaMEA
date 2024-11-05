import numpy as np

class EMODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 10
        cr = 0.9
        f = 0.8
        bounds = (-5.0, 5.0)
        
        def clip(x):
            return np.clip(x, bounds[0], bounds[1])
        
        def initialize_population():
            return np.random.uniform(bounds[0], bounds[1], size=(population_size, self.dim))
        
        def differential_evolution(population):
            new_population = np.zeros_like(population)
            for i in range(population_size):
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
            offspring = differential_evolution(population)
            for ind in offspring:
                fitness = func(ind)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            population = np.vstack((population, offspring))
        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution