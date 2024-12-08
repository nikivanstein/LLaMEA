import numpy as np

class EnhancedHybridFireflyDE(ImprovedEnhancedHybridFireflyDE):
    def __init__(self, budget, dim, alpha=0.5, beta_min=0.2, gamma=0.5, pop_size=20, adapt_rate=0.1):
        super().__init__(budget, dim, alpha, beta_min, gamma, pop_size, adapt_rate)

    def __call__(self, func):
        def cauchy_flight():
            sigma = 0.1
            v = np.random.standard_cauchy(self.dim)
            step = sigma * v / np.abs(v)
            return step

        while budget_used < self.budget:
            new_pop = []
            for i in range(self.pop_size):
                beta = self.beta_min + (1 - self.beta_min) * np.random.rand()
                attractor = pop[best_idx] + beta * (pop[i] - pop[best_idx]) + levy_flight() * chaotic_map(pop[i][chaos_idx] * self.chaos_param) + cauchy_flight()
                trial = clipToBounds(attractor)

                ...
