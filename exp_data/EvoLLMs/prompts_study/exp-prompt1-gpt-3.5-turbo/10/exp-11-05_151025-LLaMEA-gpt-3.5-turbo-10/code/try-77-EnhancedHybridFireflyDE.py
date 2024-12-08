import numpy as np

class EnhancedHybridFireflyDE(HybridFireflyDE):
    def __init__(self, budget, dim, alpha=0.5, beta_min=0.2, gamma=0.5, pop_size=20, adapt_rate=0.1):
        super().__init__(budget, dim, alpha, beta_min, gamma, pop_size)
        self.adapt_rate = adapt_rate
        self.chaos_param = 0.1

    def __call__(self, func):
        def chaotic_map(x):
            return 4 * x * (1 - x)

        def adaptive_mutation(fitness, x_r1, x_r2, x_r3, F=0.5):
            fitness_factor = np.mean(fitness) / (np.min(fitness) + 1e-8)
            F *= 1 + fitness_factor
            return x_r1 + F * (x_r2 - x_r3)

        def levy_flight():
            sigma1 = (np.math.gamma(1 + self.alpha) * np.math.sin(np.pi * self.alpha / 2) / (np.math.gamma((1 + self.alpha) / 2) * self.alpha * 2 ** ((self.alpha - 1) / 2))) ** (1 / self.alpha)
            sigma2 = 1
            u = np.random.normal(0, sigma1, self.dim)
            v = np.random.normal(0, sigma2, self.dim)
            step = u / np.power(np.abs(v), 1 / self.alpha)
            return step

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
                attractor = pop[best_idx] + beta * (pop[i] - pop[best_idx]) + levy_flight() * chaotic_map(pop[i][chaos_idx] * self.chaos_param)
                trial = clipToBounds(attractor)

                x_r1, x_r2, x_r3 = pop[np.random.choice(range(self.pop_size), 3, replace=False)]
                F = 0.5 + np.random.normal(0, self.adapt_rate)
                mutant = adaptive_mutation(fitness, x_r1, x_r2, x_r3, F)
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