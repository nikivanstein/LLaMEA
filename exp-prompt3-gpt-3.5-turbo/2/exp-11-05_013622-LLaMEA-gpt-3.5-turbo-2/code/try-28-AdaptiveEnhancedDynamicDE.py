import numpy as np

class AdaptiveEnhancedDynamicDE(EnhancedDynamicDE):
    def mutation(self, population, F, diversity):
        rand1, rand2, rand3 = np.random.randint(0, len(population), 3)
        scaling_factor = 0.5 + 0.5 * (1 - np.tanh(np.mean([func(x) for x in population])))
        mutant = population[rand1] + scaling_factor * F * (population[rand2] - population[rand3]) + np.random.standard_cauchy(self.dim) * diversity
        return np.clip(mutant, -5.0, 5.0)