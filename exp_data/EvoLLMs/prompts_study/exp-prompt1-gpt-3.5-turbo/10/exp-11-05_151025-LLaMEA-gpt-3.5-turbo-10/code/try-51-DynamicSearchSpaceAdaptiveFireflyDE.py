import numpy as np

class DynamicSearchSpaceAdaptiveFireflyDE(ImprovedEnhancedHybridFireflyDE):
    def __init__(self, budget, dim, alpha=0.5, beta_min=0.2, gamma=0.5, pop_size=20, adapt_rate=0.1):
        super().__init__(budget, dim, alpha, beta_min, gamma, pop_size, adapt_rate)
        self.initial_search_range = 5.0

    def __call__(self, func):
        def adaptive_search_range(step):
            return self.initial_search_range * np.exp(-step / self.budget)

        def clipToBounds(x, search_range):
            return np.clip(x, -search_range, search_range)

        pop = np.random.uniform(-self.initial_search_range, self.initial_search_range, (self.pop_size, self.dim))
        fitness = [func(ind) for ind in pop]
        best_idx = np.argmin(fitness)
        best_sol = pop[best_idx]
        budget_used = self.pop_size
        adapt_count = 0
        chaos_idx = np.random.randint(self.dim)

        while budget_used < self.budget:
            new_pop = []
            for i in range(self.pop_size):
                beta = self.beta_min + (1 - self.beta_min) * np.random.rand()
                search_range = adaptive_search_range(budget_used)
                attractor = pop[best_idx] + beta * (pop[i] - pop[best_idx]) + levy_flight() * chaotic_map(pop[i][chaos_idx] * self.chaos_param)
                trial = clipToBounds(attractor, search_range)

                x_r1, x_r2, x_r3 = pop[np.random.choice(range(self.pop_size), 3, replace=False)]
                F = 0.5 + np.random.normal(0, self.adapt_rate)
                mutant = de_mutate(x_r1, x_r2, x_r3, F)
                mutant = clipToBounds(mutant, search_range)

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

        return best_sol