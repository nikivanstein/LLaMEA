import numpy as np

class ImprovedDynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.CR = 0.5  # Crossover rate
        self.F_min = 0.2  # Minimum scaling factor
        self.F_max = 0.8  # Maximum scaling factor

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness_values = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            F = self.F_min + np.random.rand(self.budget) * (self.F_max - self.F_min)
            abc = np.random.choice(population, (self.budget, 3), replace=False)
            j_rand = np.random.randint(self.dim, size=self.budget)
            mutants = np.clip(abc[:, 0] + F[:, np.newaxis] * (abc[:, 1] - abc[:, 2]), -5.0, 5.0)
            
            trials = np.where(np.random.rand(self.budget, self.dim) < self.CR, mutants, population)
            trials[np.arange(self.budget), j_rand] = mutants[np.arange(self.budget), j_rand]
            
            f_trials = np.array([func(trial) for trial in trials])
            improve_mask = f_trials < fitness_values
            population[improve_mask] = trials[improve_mask]
            fitness_values[improve_mask] = f_trials[improve_mask]
        
        best_index = np.argmin(fitness_values)
        best_solution = population[best_index]
        return best_solution