class AdaptiveStepSizeDEImprovedFastConvergence(AdaptiveStepSizeDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_factor = 0.5

    def mutation(self, target, population, diversity, fitness, iteration):
        diversity_factor = np.mean(diversity) if np.mean(diversity) != 0 else 0.01
        self.F = self.mutation_factor * (1.0 / diversity_factor)
        return super().mutation(target, population, diversity, fitness, iteration)