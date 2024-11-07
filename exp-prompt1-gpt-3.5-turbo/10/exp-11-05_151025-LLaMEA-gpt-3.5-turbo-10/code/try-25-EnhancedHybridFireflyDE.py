import numpy as np

class EnhancedHybridFireflyDE(HybridFireflyDE):
    def __init__(self, budget, dim, alpha=0.5, beta_min=0.2, gamma=0.5, pop_size=20, adapt_rate=0.1):
        super().__init__(budget, dim, alpha, beta_min, gamma, pop_size)
        self.adapt_rate = adapt_rate
        self.max_pop_size = pop_size * 2
        self.min_pop_size = pop_size // 2

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
        adapt_count = 0
        adapt_threshold = 20
        
        while budget_used < self.budget:
            new_pop = []
            for i in range(self.pop_size):
                beta = self.beta_min + (1 - self.beta_min) * np.random.rand()
                attractor = pop[best_idx] + beta * (pop[i] - pop[best_idx]) + levy_flight()
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
                    
                if adapt_count >= adapt_threshold:  # Adaptive mutation strategy update
                    self.adapt_rate *= 0.85
                    adapt_threshold += 10
                    adapt_count = 0
                    
            if adapt_count > 10:  # Dynamic population size adjustment
                self.pop_size = min(self.max_pop_size, self.pop_size + 1)
                adapt_count = 0
            elif adapt_count < 5 and self.pop_size > self.min_pop_size:
                self.pop_size = max(self.min_pop_size, self.pop_size - 1)
                adapt_count = 0
                
            pop = np.array(new_pop)
            fitness = [func(ind) for ind in pop]
            best_idx = np.argmin(fitness)
            best_sol = pop[best_idx]

        return best_sol