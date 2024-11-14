import numpy as np

class EnhancedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.weight_factor = 0.5

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                a, b, c = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant = population[a] + self.scale_factor * (population[b] - population[c]) * (1 + self.weight_factor * (fitness[a] - fitness[b]))
                for j in range(self.dim):
                    if np.random.rand() > self.crossover_rate:
                        mutant[j] = population[i][j]
                mutant_fit = func(mutant)
                if mutant_fit < fitness[i]:
                    population[i] = mutant
                    fitness[i] = mutant_fit
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        return best_solution, best_fitness