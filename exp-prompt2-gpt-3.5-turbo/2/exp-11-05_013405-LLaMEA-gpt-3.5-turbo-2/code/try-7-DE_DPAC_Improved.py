import numpy as np

class DE_DPAC_Improved:
    def __init__(self, budget, dim, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR

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
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, x)
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    success_rates[i] += 1
                else:
                    success_rates[i] -= 1
            
            # Update F and CR based on the success rates
            success_rates[success_rates < 0] = 0.1
            self.F = np.clip(np.mean(success_rates) / 10, 0.1, 0.9)
            self.CR = np.clip((1 - np.mean(success_rates)) + np.random.normal(0, 0.1), 0.1, 1.0)
            
            # Dynamic population size adaptation based on success rates
            if np.random.rand() < 0.1:
                successful_inds = np.where(success_rates > np.mean(success_rates))[0]
                failed_inds = np.where(success_rates <= np.mean(success_rates))[0]
                if len(successful_inds) > len(failed_inds):
                    if self.pop_size < 100:
                        self.pop_size += 1
                elif len(successful_inds) < len(failed_inds):
                    if self.pop_size > 10:
                        self.pop_size -= 1
                pop = np.vstack([pop[successful_inds], np.random.uniform(-5.0, 5.0, (self.pop_size - len(successful_inds), self.dim))])
                fitness = np.array([func(ind) for ind in pop])
                success_rates = np.zeros(self.pop_size)
        
        return pop[np.argmin(fitness)]