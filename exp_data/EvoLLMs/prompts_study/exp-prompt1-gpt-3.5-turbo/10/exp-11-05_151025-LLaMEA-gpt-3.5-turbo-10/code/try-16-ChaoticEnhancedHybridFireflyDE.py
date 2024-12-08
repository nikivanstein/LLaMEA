import numpy as np

class ChaoticEnhancedHybridFireflyDE(EnhancedHybridFireflyDE):
    def levy_flight(self):
        chaos_map = lambda x: 3.9 * x * (1 - x)  # Logistic chaotic map
        step = np.zeros(self.dim)
        x = np.random.rand(self.dim)
        for _ in range(self.dim):
            x = chaos_map(x)
            step[_] = 10 * (x - 0.5)  # Scale chaotic value to search space
        return step

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = [func(ind) for ind in pop]
        best_idx = np.argmin(fitness)
        best_sol = pop[best_idx]
        budget_used = self.pop_size
        adapt_count = 0
        
        while budget_used < self.budget:
            new_pop = []
            for i in range(self.pop_size):
                beta = self.beta_min + (1 - self.beta_min) * np.random.rand()
                attractor = pop[best_idx] + beta * (pop[i] - pop[best_idx]) + self.levy_flight()
                trial = self.clipToBounds(attractor)
                
                x_r1, x_r2, x_r3 = pop[np.random.choice(range(self.pop_size), 3, replace=False)]
                F = 0.5 + np.random.normal(0, self.adapt_rate)
                mutant = self.de_mutate(x_r1, x_r2, x_r3, F)
                mutant = self.clipToBounds(mutant)
                
                if func(mutant) < func(trial):
                    new_pop.append(mutant)
                    budget_used += 1
                    adapt_count += 1
                else:
                    new_pop.append(trial)
                    budget_used += 1
                    
                if adapt_count >= 10:
                    self.adapt_rate *= 0.9
                    adapt_count = 0
                    
            pop = np.array(new_pop)
            fitness = [func(ind) for ind in pop]
            best_idx = np.argmin(fitness)
            best_sol = pop[best_idx]

        return best_sol