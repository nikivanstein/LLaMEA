class AdaptiveStepSizeDEImproved(AdaptiveStepSizeDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_factor = 0.5

    def mutation(self, target, population, diversity, fitness, iteration):
        fitness_diff = np.mean(np.abs(np.diff(fitness)))
        dynamic_factor = np.clip(1.0 / (1.0 + fitness_diff), 0.1, 0.9)
        self.F = self.mutation_factor * dynamic_factor
        return super().mutation(target, population, diversity, fitness, iteration)