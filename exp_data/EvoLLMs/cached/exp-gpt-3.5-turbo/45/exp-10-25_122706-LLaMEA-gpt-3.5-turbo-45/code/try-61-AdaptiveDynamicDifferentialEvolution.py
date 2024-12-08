import numpy as np

class AdaptiveDynamicDifferentialEvolution(DynamicDifferentialEvolution):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.adaptive_F = np.full(self.pop_size, 0.5)
        self.adaptive_CR = np.full(self.pop_size, 0.9)

    def __call__(self, func):
        for _ in range(self.budget):
            trial_population = np.zeros_like(self.population)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mut = np.clip(np.random.normal(self.adaptive_F[i], 0.1), self.min_mut, self.max_mut)
                cross = np.clip(np.random.normal(self.adaptive_CR[i], 0.1), self.min_cross, self.max_cross)
                trial = a + mut * (b - c)
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < cross or j == j_rand:
                        trial_population[i, j] = trial[j]
                    else:
                        trial_population[i, j] = self.population[i, j]
                if func(trial_population[i]) < func(self.population[i]):
                    self.population[i] = trial_population[i]
                    self.adaptive_F[i] = np.clip(self.adaptive_F[i] + np.random.normal(0.0, 0.1), self.min_mut, self.max_mut)
                    self.adaptive_CR[i] = np.clip(self.adaptive_CR[i] + np.random.normal(0.0, 0.1), self.min_cross, self.max_cross)
        return self.population[np.argmin([func(ind) for ind in self.population])]