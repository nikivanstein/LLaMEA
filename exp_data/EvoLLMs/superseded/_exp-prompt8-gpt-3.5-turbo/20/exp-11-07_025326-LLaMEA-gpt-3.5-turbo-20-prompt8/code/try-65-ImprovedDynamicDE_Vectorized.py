import numpy as np

class ImprovedDynamicDE_Vectorized:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.CR = 0.5  # Crossover rate
        self.F_min = 0.2  # Minimum scaling factor
        self.F_max = 0.8  # Maximum scaling factor

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (int(0.8*self.budget), self.dim))  # Reduced population initialization size
        fitness_values = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            F = self.F_min + np.random.rand() * (self.F_max - self.F_min)
            idx = np.arange(len(population))
            np.random.shuffle(idx)
            
            a, b, c = population[np.random.choice(idx, (3, len(population)), replace=True)]
            j_rand = np.random.randint(self.dim)
            mutants = np.clip(a + F * (b - c), -5.0, 5.0)
            
            trials = np.where(np.random.rand(len(population), self.dim) < self.CR, mutants, population)
            f_trials = np.array([func(trial) for trial in trials])
            
            improve_mask = f_trials < fitness_values
            population[improve_mask] = trials[improve_mask]
            fitness_values[improve_mask] = f_trials[improve_mask]

        best_index = np.argmin(fitness_values)
        best_solution = population[best_index]
        return best_solution