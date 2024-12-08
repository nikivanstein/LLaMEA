import numpy as np

class AdaptiveMutationPSODE(DynamicHybridPSODE):
    def de_update(self, population, func):
        for i in range(self.population_size):
            x, a, b, c = population[i]['position'], population[np.random.randint(self.population_size)]['position'], population[np.random.randint(self.population_size)]['position'], population[np.random.randint(self.population_size)]['position']
            mutant = np.clip(a + self.f * (b - c), -5.0, 5.0)
            trial = np.where(np.random.rand(self.dim) <= self.cr, mutant, x)
            if func(trial) < func(x):
                population[i]['position'] = trial.copy()
                self.f = np.clip(np.random.normal(self.f, 0.1), self.min_f, self.max_f)  # Adaptive mutation rate
        return population