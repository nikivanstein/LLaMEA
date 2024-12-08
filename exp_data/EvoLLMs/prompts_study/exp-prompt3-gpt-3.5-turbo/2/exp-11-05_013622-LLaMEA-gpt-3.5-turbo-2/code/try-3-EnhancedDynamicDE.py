import numpy as np

class EnhancedDynamicDE(DynamicDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        CR = 0.9
        for _ in range(self.budget):
            new_population = []
            for idx, candidate in enumerate(population):
                F = 0.5 + 0.3 * np.tanh(0.01 * func(candidate))
                mutant = self.mutation(population, F)
                crossover_points = np.random.rand(self.dim) < CR
                trial = np.where(crossover_points, mutant, candidate)
                if func(trial) < func(candidate):
                    new_population.append(trial)
                else:
                    new_population.append(candidate)
            population = np.array(new_population)
        return population[np.argmin([func(candidate) for candidate in population])]