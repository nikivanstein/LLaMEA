import numpy as np

class AdaptiveRefinedEnhancedDynamicDE(EnhancedDynamicDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.F_lb = 0.5
        self.F_ub = 1.0

    def mutation(self, population, F, diversity):
        rand1, rand2, rand3 = np.random.randint(0, len(population), 3)
        F_factor = self.F_lb + (self.F_ub - self.F_lb) * np.random.rand()
        mutant = population[rand1] + F_factor * (population[rand2] - population[rand3]) + np.random.standard_cauchy(self.dim) * diversity
        return np.clip(mutant, -5.0, 5.0)