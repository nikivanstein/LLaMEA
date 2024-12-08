import numpy as np

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.F = 0.5
        self.CR = 0.9
        self.min_mut = 0.2
        self.max_mut = 0.8
        self.min_cross = 0.5
        self.max_cross = 1.0
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))

    def __call__(self, func):
        for _ in range(self.budget):
            trial_population = np.zeros_like(self.population)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mut = np.clip(np.random.normal(self.F, 0.1), self.min_mut, self.max_mut)
                cross = np.clip(np.random.normal(self.CR, 0.1), self.min_cross, self.max_cross)
                trial = a + mut * (b - c)
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < cross or j == j_rand:
                        trial_population[i, j] = trial[j]
                    else:
                        trial_population[i, j] = self.population[i, j]
                if func(trial_population[i]) < func(self.population[i]):
                    self.population[i] = trial_population[i]
        return self.population[np.argmin([func(ind) for ind in self.population])]