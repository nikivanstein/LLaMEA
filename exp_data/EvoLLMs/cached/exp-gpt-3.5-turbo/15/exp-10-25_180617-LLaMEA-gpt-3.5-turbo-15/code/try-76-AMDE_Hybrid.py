import numpy as np

class AMDE_Hybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.max_iter = budget // self.pop_size

    def __call__(self, func):
        lb = -5.0
        ub = 5.0
        pop = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        vel = np.zeros((self.pop_size, self.dim))
        pbest = pop.copy()
        pbest_fit = np.array([func(ind) for ind in pbest])
        gbest = pbest[pbest_fit.argmin()]
        gbest_fit = pbest_fit.min()

        f = 0.5  # Scale factor for adaptive mutation

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                trial = pop[i] + f * (pbest[i] - pop[i]) + f * (pop[np.random.choice(self.pop_size)] - pop[np.random.choice(self.pop_size)])
                trial = np.clip(trial, lb, ub)
                fit = func(trial)
                if fit < pbest_fit[i]:
                    pbest[i] = trial
                    pbest_fit[i] = fit
                    if fit < gbest_fit:
                        gbest = trial
                        gbest_fit = fit

        return gbest