import numpy as np

class EnhancedDynamicControlEMODE_Tournament:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.cr = 0.9
        self.f = 0.8
        self.scale = 0.1
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        
        def differential_evolution(population, population_size, f, cr):
            new_population = np.zeros_like(population)
            fitness_values = [func(ind) for ind in population]
            selected_population = self.tournament_selection(population, fitness_values, population_size)
            for i in range(population_size):
                target = selected_population[i]
                candidates = np.delete(selected_population, i, axis=0)
                r1, r2, r3 = candidates[np.random.choice(range(len(candidates)), 3, replace=False)]
                noise = np.random.standard_cauchy(self.dim) * self.scale
                mutant = self.clip(r1 + f * (r2 - r3) + noise)
                crossover_points = np.random.rand(self.dim) < cr
                offspring = np.where(crossover_points, mutant, target)
                new_population[i] = self.clip(offspring)
            return new_population
        
        evaluations = 0
        population_size = 10
        population = self.initialize_population(population_size)
        while evaluations < self.budget:
            self.f = max(0.4, self.f - 0.0005)
            self.cr = min(0.95, self.cr + 0.0005)
            
            offspring = differential_evolution(population, population_size, self.f, self.cr)
            for ind in offspring:
                fitness = func(ind)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            population = np.vstack((population, offspring))
            population_size = min(50, int(population_size * 1.1))
        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution