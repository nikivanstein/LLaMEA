import numpy as np

class EnhancedFireflyDEAlgorithm(FireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.cr = 0.5
        self.f = 0.5

    def differential_evolution(self, i, t):
        r1, r2, r3 = np.random.choice(self.population, 3, replace=False)
        mutant = self.population[r1] + self.f * (self.population[r2] - self.population[r3])
        trial = np.copy(self.population[i])
        for j in range(self.dim):
            if np.random.rand() < self.cr or j == np.random.randint(self.dim):
                trial[j] = mutant[j]
        return trial

    def __call__(self, func):
        for t in range(self.budget):
            for i in range(self.budget):
                trial = self.differential_evolution(i, t)
                if func(trial) < func(self.population[i]):
                    self.population[i] = trial
            best_index = np.argmin([func(ind) for ind in self.population])
            for i in range(self.dim):
                self.population[best_index][i] = np.clip(self.population[best_index][i], self.lower_bound, self.upper_bound)

        return self.population[np.argmin([func(ind) for ind in self.population])]
        