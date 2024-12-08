import numpy as np

class AdaptiveDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.scale_factors = np.full(self.population_size, self.scale_factor)

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                a, b, c = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                scale_factor = np.clip(self.scale_factors[i], 0.1, 0.9)  # Adaptive scale factor
                mutant = population[a] + scale_factor * (population[b] - population[c])
                for j in range(self.dim):
                    if np.random.rand() > self.crossover_rate:
                        mutant[j] = population[i][j]
                mutant_fit = func(mutant)
                if mutant_fit < fitness[i]:
                    population[i] = mutant
                    fitness[i] = mutant_fit
                    self.scale_factors[i] *= 1.1  # Increase scale factor on improvement
                else:
                    self.scale_factors[i] *= 0.9  # Decrease scale factor on no improvement
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        return best_solution, best_fitness