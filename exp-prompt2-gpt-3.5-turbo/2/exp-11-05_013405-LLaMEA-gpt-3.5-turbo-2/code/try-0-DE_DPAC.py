import numpy as np

class DE_DPAC:
    def __init__(self, budget, dim, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
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
        return pop[np.argmin(fitness)]