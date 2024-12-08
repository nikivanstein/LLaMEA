class AdaptiveMutationDEImproved(AdaptiveStepSizeDEImproved):
    def mutation(self, target, population, diversity, fitness, iteration):
        diversity_factor = np.clip(1.0 / (1.0 + diversity), 0.1, 0.9)
        self.F = self.mutation_factor * diversity_factor
        return super().mutation(target, population, diversity, fitness, iteration)