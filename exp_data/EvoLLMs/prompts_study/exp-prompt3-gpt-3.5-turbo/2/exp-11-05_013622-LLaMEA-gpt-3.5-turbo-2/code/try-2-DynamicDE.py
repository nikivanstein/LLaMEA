import numpy as np

class DynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def mutation(self, population, F):
        rand1, rand2, rand3 = np.random.randint(0, len(population), 3)
        mutant = population[rand1] + F * (population[rand2] - population[rand3])
        return np.clip(mutant, -5.0, 5.0)
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        F = 0.5
        CR = 0.9
        for _ in range(self.budget):
            new_population = []
            for idx, candidate in enumerate(population):
                mutant = self.mutation(population, F)
                crossover_points = np.random.rand(self.dim) < CR
                trial = np.where(crossover_points, mutant, candidate)
                if func(trial) < func(candidate):
                    new_population.append(trial)
                else:
                    new_population.append(candidate)
            population = np.array(new_population)
        return population[np.argmin([func(candidate) for candidate in population])]