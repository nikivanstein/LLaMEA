class EnhancedLevyFlightEvolutionaryDifferentialHarmonySearch:
    def __call__(self, func):
        def levy_flight_mutation(x, lambda_=1.5, alpha=0.01):
            levy = lambda_ * np.random.standard_cauchy(size=self.dim) / (np.abs(np.random.normal(0, 1, size=self.dim)) ** (1/alpha))
            mutant = x + levy
            return np.clip(mutant, -5.0, 5.0)

        for _ in range(self.budget):
            for i in range(self.budget):
                x = population[i]
                F = F_adapt + 0.1 * np.random.randn()
                F = np.clip(F, F_lower, F_upper)

                mutant = levy_flight_mutation(x)
                trial = crossover(x, mutant, CR)
                
                population[i] = trial if cost_function(trial) < cost_function(x) else harmonic_search(x)