import numpy as np

class ImprovedEnhancedDynamicDE(EnhancedDynamicDE):
    def mutation(self, population, F, diversity):
        rand1, rand2, rand3 = np.random.randint(0, len(population), 3)
        differential_weight = np.random.uniform(0, 2)
        mutant = population[rand1] + differential_weight * F * (population[rand2] - population[rand3]) + np.random.standard_cauchy(self.dim) * diversity
        return np.clip(mutant, -5.0, 5.0)