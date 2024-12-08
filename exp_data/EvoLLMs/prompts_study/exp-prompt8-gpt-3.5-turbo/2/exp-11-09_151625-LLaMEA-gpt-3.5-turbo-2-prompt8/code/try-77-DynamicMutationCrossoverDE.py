import numpy as np

class DynamicMutationCrossoverDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.CR_min = 0.1  # Minimum crossover rate
        self.CR_max = 0.9  # Maximum crossover rate
        self.F_min = 0.1  # Minimum mutation factor
        self.F_max = 0.9  # Maximum mutation factor

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            target = population[i]
            idxs = np.arange(self.budget)
            np.random.shuffle(idxs)
            a, b, c = population[np.random.choice(idxs[:3], 3, replace=False)]
            F = np.random.uniform(self.F_min, self.F_max)
            CR = np.random.uniform(self.CR_min, self.CR_max)
            mutant = a + F * (b - c)
            crossover = np.random.rand(self.dim) < CR
            trial = np.where(crossover, mutant, target)
            
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                if np.random.rand() < 0.1:  # Introduce probabilistic mutation
                    self.F_min = np.clip(np.random.normal(self.F_min, 0.05), 0.1, 0.9)
                    self.F_max = np.clip(np.random.normal(self.F_max, 0.05), 0.1, 0.9)
                    self.CR_min = np.clip(np.random.normal(self.CR_min, 0.05), 0.1, 0.9)
                    self.CR_max = np.clip(np.random.normal(self.CR_max, 0.05), 0.1, 0.9)
        best_idx = np.argmin(fitness)
        return population[best_idx]