class DynamicScalingRefinedEnhancedDynamicDE(EnhancedDynamicDE):
    def mutation(self, population, F, diversity):
        rand1, rand2, rand3 = np.random.randint(0, len(population), 3)
        fitness_improvement = np.mean([func(ind) - self.archive_fitness[ind] for ind in population])
        scaling_factor = np.exp(-fitness_improvement)
        mutant = population[rand1] + F * scaling_factor * (population[rand2] - population[rand3]) + np.random.standard_cauchy(self.dim) * diversity
        return np.clip(mutant, -5.0, 5.0)