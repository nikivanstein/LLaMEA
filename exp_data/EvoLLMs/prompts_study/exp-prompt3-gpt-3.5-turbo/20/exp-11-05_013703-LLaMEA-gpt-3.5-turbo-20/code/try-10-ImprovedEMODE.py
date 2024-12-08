import numpy as np

class ImprovedEMODE(EMODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.adaptive_cr = 0.9
        self.adaptive_f = 0.8

    def __call__(self, func):
        population_size = 10
        bounds = (-5.0, 5.0)
        
        def differential_evolution(population):
            nonlocal self.adaptive_cr, self.adaptive_f
            new_population = np.zeros_like(population)
            for i in range(population_size):
                target = population[i]
                r1, r2, r3 = np.random.choice(population, 3, replace=False)
                current_cr = np.clip(self.adaptive_cr + np.random.normal(0, 0.1), 0, 1)
                current_f = np.clip(self.adaptive_f + np.random.normal(0, 0.1), 0, 2)
                mutant = clip(r1 + current_f * (r2 - r3))
                crossover_points = np.random.rand(self.dim) < current_cr
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