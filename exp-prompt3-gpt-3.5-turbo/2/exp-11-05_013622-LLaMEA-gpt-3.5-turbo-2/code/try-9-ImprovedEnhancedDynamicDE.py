import numpy as np

class ImprovedEnhancedDynamicDE(EnhancedDynamicDE):
    def mutation(self, population, F, diversity):
        rand1, rand2, rand3 = np.random.randint(0, len(population), 3)
        cauchy_perturbation = np.random.standard_cauchy(self.dim)
        gaussian_perturbation = np.random.normal(0, 1, self.dim)
        mutant = population[rand1] + F * (population[rand2] - population[rand3]) + 0.5 * (cauchy_perturbation + gaussian_perturbation) * diversity
        return np.clip(mutant, -5.0, 5.0)