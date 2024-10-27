# import numpy as np

class RefinedDynamicDifferentialEvolution(DynamicDifferentialEvolution):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        
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
                self.F = np.clip(self.F + np.random.normal(0.0, 0.1), self.min_mut, self.max_mut)
                self.CR = np.clip(self.CR + np.random.normal(0.0, 0.1), self.min_cross, self.max_cross)
                self.CR = np.clip(self.CR + np.random.normal(0.0, 0.1), self.min_cross, self.max_cross)  # Probabilistic change
                
        return self.population[np.argmin([func(ind) for ind in self.population])]