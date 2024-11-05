import numpy as np

class OpposedEnhancedDynamicDE(EnhancedDynamicDE):
    def mutation(self, population, F, diversity):
        rand1, rand2, rand3 = np.random.randint(0, len(population), 3)
        mutant = population[rand1] + F * (population[rand2] - population[rand3]) + np.random.standard_cauchy(self.dim) * diversity
        opposite_mutant = -mutant
        return np.clip(mutant + opposite_mutant, -5.0, 5.0)