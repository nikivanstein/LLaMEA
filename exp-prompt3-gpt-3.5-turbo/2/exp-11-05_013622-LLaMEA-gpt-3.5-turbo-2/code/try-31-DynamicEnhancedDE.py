import numpy as np

class DynamicEnhancedDE(EnhancedDynamicDE):
    def mutation(self, population, F, diversity):
        rand1, rand2, rand3 = np.random.randint(0, len(population), 3)
        mutant = population[rand1] + F * (population[rand2] - population[rand3]) + np.random.standard_cauchy(self.dim) * diversity + np.random.normal(0, 1, self.dim) * diversity
        return np.clip(mutant, -5.0, 5.0)