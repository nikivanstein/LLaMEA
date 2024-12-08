import numpy as np

class EnhancedRefinedAdaptiveMutationPSODE(RefinedAdaptiveMutationPSODE):
    def de_update(self, population, func):
        for i in range(self.population_size):
            x, a, b, c = population[i]['position'], population[np.random.randint(self.population_size)]['position'], population[np.random.randint(self.population_size)]['position'], population[np.random.randint(self.population_size)]['position']
            if np.random.rand() < 0.2:
                self.f = np.clip(np.random.normal(self.f, 0.1), self.min_f, self.max_f)  # Probabilistic mutation rate change
                line_to_change = np.random.randint(self.dim)  # Introducing line selection for mutation
                x[line_to_change] = np.clip(np.random.normal(x[line_to_change], 0.1), -5.0, 5.0)  # Probabilistic line change
            mutant = np.clip(a + self.f * (b - c), -5.0, 5.0)
            trial = np.where(np.random.rand(self.dim) <= self.cr, mutant, x)
            if func(trial) < func(x):
                population[i]['position'] = trial.copy()
        return population