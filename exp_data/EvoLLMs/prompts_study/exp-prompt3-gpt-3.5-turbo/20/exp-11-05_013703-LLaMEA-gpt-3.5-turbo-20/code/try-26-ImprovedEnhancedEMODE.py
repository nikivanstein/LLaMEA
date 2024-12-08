import numpy as np

class ImprovedEnhancedEMODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        cr = 0.9
        f = 0.8
        scale = 0.1
        bounds = (-5.0, 5.0)
        
        def clip(x):
            return np.clip(x, bounds[0], bounds[1])
        
        def initialize_population(population_size):
            return np.random.uniform(bounds[0], bounds[1], size=(population_size, self.dim))
        
        def differential_evolution(population, population_size):
            new_population = np.zeros_like(population)
            for i in range(population_size):
                target = population[i]
                candidates = np.delete(population, i, axis=0)
                r1, r2, r3 = candidates[np.random.choice(range(len(candidates)), 3, replace=False)]
                noise = np.random.standard_cauchy(self.dim) * scale
                levy = np.random.standard_cauchy(self.dim) / np.sqrt(np.random.gamma(1.5, 2.0, self.dim))
                mutant = clip(r1 + f * (r2 - r3) + noise + levy)
                crossover_points = np.random.rand(self.dim) < cr
                offspring = np.where(crossover_points, mutant, target)
                new_population[i] = clip(offspring)
            return new_population
        
        evaluations = 0
        population_size = 10
        population = initialize_population(population_size)
        while evaluations < self.budget:
            offspring = differential_evolution(population, population_size)
            for ind in offspring:
                fitness = func(ind)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            population = np.vstack((population, offspring))
            population_size = min(50, int(population_size * 1.1))  # Adaptive population size increase
        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution