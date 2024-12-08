import numpy as np

class DE_DPAC_Improved_Diversity:
    def __init__(self, budget, dim, pop_size=50, F=0.5, CR=0.9, alpha=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.alpha = alpha

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        success_rates = np.zeros(self.pop_size)
        
        for _ in range(self.budget - self.pop_size):
            F = np.random.uniform(0, 1) if np.random.rand() > 0.1 else self.F
            CR = np.random.normal(self.CR, 0.1)
            idx = np.arange(self.pop_size)
            np.random.shuffle(idx)
            for i, x in enumerate(pop):
                a, b, c = pop[np.random.choice(idx[:3], 3, replace=False)]
                div_factor = np.mean(np.abs(pop - x), axis=0)
                mutant = np.clip(a + F * (b - c) + self.alpha * div_factor * np.random.randn(self.dim), -5.0, 5.0)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, x)
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    success_rates[i] += 1
                else:
                    success_rates[i] -= 1
            
            success_rates[success_rates < 0] = 0.1
            self.F = np.clip(np.mean(success_rates) / 10, 0.1, 0.9)
            self.CR = np.clip((1 - np.mean(success_rates)) + np.random.normal(0, 0.1), 0.1, 1.0)
        
        return pop[np.argmin(fitness)]