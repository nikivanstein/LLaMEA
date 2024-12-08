import numpy as np

class ImprovedEnhancedDynamicDE(EnhancedDynamicDE):
    def mutation(self, population, F, diversity):
        rand1, rand2, rand3, rand4, rand5 = np.random.randint(0, len(population), 5)
        mutant = population[rand1] + F * (population[rand2] - population[rand3]) + np.random.normal(0, 1, self.dim) * diversity + np.random.normal(0, 1, self.dim) * (population[rand4] - population[rand5])
        return np.clip(mutant, -5.0, 5.0)