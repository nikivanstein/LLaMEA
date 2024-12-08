import numpy as np

class ImprovedRefinedEnhancedDynamicDE(EnhancedDynamicDE):
    def mutation(self, population, F, diversity):
        rand1, rand2, rand3 = np.random.randint(0, len(population), 3)
        levy = np.random.standard_cauchy(self.dim)
        mutant = population[rand1] + F * (population[rand2] - population[rand3]) + levy * diversity
        return np.clip(mutant, -5.0, 5.0)