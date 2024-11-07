import numpy as np

class DynamicPopSizeHybridFireflyDE(HybridFireflyDE):
    def __init__(self, budget, dim, alpha=0.5, beta_min=0.2, gamma=0.5, pop_size=20, max_pop_size=30, min_pop_size=10):
        super().__init__(budget, dim, alpha, beta_min, gamma, pop_size)
        self.max_pop_size = max_pop_size
        self.min_pop_size = min_pop_size

    def __call__(self, func):
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

        while budget_used < self.budget:
            new_pop = []
            for i in range(self.pop_size):
                beta = self.beta_min + (1 - self.beta_min) * np.random.rand()
                attractor = pop[best_idx] + beta * (pop[i] - pop[best_idx]) + levy_flight()
                trial = clipToBounds(attractor)

                x_r1, x_r2, x_r3 = pop[np.random.choice(range(self.pop_size), 3, replace=False)]
                mutant = de_mutate(x_r1, x_r2, x_r3)
                mutant = clipToBounds(mutant)

                if func(mutant) < func(trial):
                    new_pop.append(mutant)
                    budget_used += 1
                else:
                    new_pop.append(trial)
                    budget_used += 1

            pop = np.array(new_pop)
            fitness = [func(ind) for ind in pop]
            best_idx = np.argmin(fitness)
            best_sol = pop[best_idx]

            # Dynamic population size adjustment
            self.pop_size = max(self.min_pop_size, min(self.max_pop_size, int(self.pop_size * (1 + self.gamma * (func(best_sol) - func(pop[best_idx])))))

        return best_sol