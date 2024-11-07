import numpy as np

class CrowdingEnhancedHybridFireflyDE(HybridFireflyDE):
    def __init__(self, budget, dim, alpha=0.5, beta_min=0.2, gamma=0.5, pop_size=20, adapt_rate=0.1, crowding_factor=0.2):
        super().__init__(budget, dim, alpha, beta_min, gamma, pop_size)
        self.adapt_rate = adapt_rate
        self.crowding_factor = crowding_factor

    def __call__(self, func):
        def chaotic_map(x):
            return 4 * x * (1 - x)

        def levy_flight():
            sigma1 = (np.math.gamma(1 + self.alpha) * np.math.sin(np.pi * self.alpha / 2) / (np.math.gamma((1 + self.alpha) / 2) * self.alpha * 2 ** ((self.alpha - 1) / 2))) ** (1 / self.alpha)
            sigma2 = 1
            u = np.random.normal(0, sigma1, self.dim)
            v = np.random.normal(0, sigma2, self.dim)
            step = u / np.power(np.abs(v), 1 / self.alpha)
            return step

        def de_mutate(x_r1, x_r2, x_r3, F=0.5):
            return x_r1 + F * (x_r2 - x_r3)

        def clipToBounds(x):
            return np.clip(x, -5.0, 5.0)

        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = [func(ind) for ind in pop]
        best_idx = np.argmin(fitness)
        best_sol = pop[best_idx]
        budget_used = self.pop_size
        adapt_count = 0

        while budget_used < self.budget:
            new_pop = []
            crowding_dist = np.zeros(self.pop_size)

            for i in range(self.pop_size):
                beta = self.beta_min + (1 - self.beta_min) * np.random.rand()
                attractor = pop[best_idx] + beta * (pop[i] - pop[best_idx]) + levy_flight() * chaotic_map(pop[i][np.random.randint(self.dim)] * self.crowding_factor)
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

                crowding_dist[i] = np.mean(np.linalg.norm(pop - pop[i], axis=1))

            sorted_indices = np.argsort(crowding_dist)
            pop = pop[sorted_indices]
            fitness = [func(ind) for ind in pop]
            best_idx = np.argmin(fitness)
            best_sol = pop[best_idx]

        return best_sol