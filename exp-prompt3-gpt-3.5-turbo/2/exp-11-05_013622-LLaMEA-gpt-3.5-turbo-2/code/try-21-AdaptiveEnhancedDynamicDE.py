import numpy as np

class AdaptiveEnhancedDynamicDE(EnhancedDynamicDE):
    def mutation(self, population, F, diversity):
        rand1, rand2, rand3 = np.random.randint(0, len(population), 3)
        cauchy_scale = np.abs(np.random.standard_cauchy(self.dim)) * np.sqrt(self.dim) / np.sqrt(self.budget)
        mutant = population[rand1] + F * (population[rand2] - population[rand3]) + cauchy_scale * diversity
        return np.clip(mutant, -5.0, 5.0)