import numpy as np

class ImprovedDynamicDE(EnhancedDynamicDE):
    def mutation(self, population, F, diversity):
        rand1, rand2, rand3 = np.random.randint(0, len(population), 3)
        cauchy_perturbation = np.random.standard_cauchy(self.dim) * diversity
        gaussian_perturbation = np.random.normal(0, 1, self.dim) * 0.1
        mutant = population[rand1] + F * (population[rand2] - population[rand3]) + cauchy_perturbation + gaussian_perturbation
        return np.clip(mutant, -5.0, 5.0)