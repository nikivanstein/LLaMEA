import numpy as np

class EnhancedHybridFireflyDE(HybridFireflyDE):
    def __init__(self, budget, dim, alpha=0.5, beta_min=0.2, gamma=0.5, pop_size=20, adapt_rate=0.1):
        super().__init__(budget, dim, alpha, beta_min, gamma, pop_size)
        self.adapt_rate = adapt_rate
        self.chaos_param = 0.1

    def __call__(self, func):
        def chaotic_map(x):
            return 4 * x * (1 - x)

        def cauchy_mutation(scale=0.1):
            return np.random.standard_cauchy(self.dim) * scale

        def clipToBounds(x):
            return np.clip(x, -5.0, 5.0)

        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
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
                attractor = pop[best_idx] + beta * (pop[i] - pop[best_idx]) + cauchy_mutation()
                trial = clipToBounds(attractor)

                x_r1, x_r2, x_r3 = pop[np.random.choice(range(self.pop_size), 3, replace=False)]
                F = 0.5 + np.random.normal(0, self.adapt_rate)
                mutant = de_mutate(x_r1, x_r2, x_r3, F)
                mutant = clipToBounds(mutant)

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