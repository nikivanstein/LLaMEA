class DynamicChaosEnhancedHybridFireflyDE(ImprovedEnhancedHybridFireflyDE):
    def __init__(self, budget, dim, alpha=0.5, beta_min=0.2, gamma=0.5, pop_size=20, adapt_rate=0.1, chaos_adapt_rate=0.05):
        super().__init__(budget, dim, alpha, beta_min, gamma, pop_size, adapt_rate)
        self.chaos_adapt_rate = chaos_adapt_rate

    def __call__(self, func):
        def clipToBounds(x):
            return np.clip(x, -5.0, 5.0)

        chaos_param = 0.1  # Initial value
        chaos_idx = np.random.randint(self.dim)

        while budget_used < self.budget:
            new_pop = []
            for i in range(self.pop_size):
                beta = self.beta_min + (1 - self.beta_min) * np.random.rand()
                attractor = pop[best_idx] + beta * (pop[i] - pop[best_idx]) + levy_flight() * chaotic_map(pop[i][chaos_idx] * chaos_param)
                trial = clipToBounds(attractor)

                if func(mutant) < func(trial):
                    new_pop.append(mutant)
                    budget_used += 1
                    adapt_count += 1
                else:
                    new_pop.append(trial)
                    budget_used += 1

                if adapt_count >= 15:
                    self.adapt_rate *= 0.9
                    adapt_count = 0

            pop = np.array(new_pop)
            fitness = [func(ind) for ind in pop]
            best_idx = np.argmin(fitness)
            best_sol = pop[best_idx]

            if adapt_count >= 10:
                chaos_param += self.chaos_adapt_rate
                chaos_param = min(chaos_param, 1.0)
                chaos_idx = np.random.randint(self.dim)

        return best_sol