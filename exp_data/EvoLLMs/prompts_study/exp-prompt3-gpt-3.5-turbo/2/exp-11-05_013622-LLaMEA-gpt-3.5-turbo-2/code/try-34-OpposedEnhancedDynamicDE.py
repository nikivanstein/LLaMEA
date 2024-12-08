import numpy as np

class OpposedEnhancedDynamicDE(EnhancedDynamicDE):
    def mutation(self, population, F, diversity):
        rand1, rand2, rand3 = np.random.randint(0, len(population), 3)
        mutant = population[rand1] + F * (population[rand2] - population[rand3]) + np.random.standard_cauchy(self.dim) * diversity
        opposite_mutant = 5.0 - mutant  # Generate opposite solution
        return np.clip(mutant, -5.0, 5.0) if func(mutant) < func(opposite_mutant) else np.clip(opposite_mutant, -5.0, 5.0)