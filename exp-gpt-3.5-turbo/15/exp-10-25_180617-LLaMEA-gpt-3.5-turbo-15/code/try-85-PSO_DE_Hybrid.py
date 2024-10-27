import numpy as np

class PSO_DE_Hybrid:
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

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                vel[i] = 0.5 * vel[i] + 2.0 * r1 * (pbest[i] - pop[i]) + 2.0 * r2 * (gbest - pop[i])
                pop[i] = np.clip(pop[i] + vel[i], lb, ub)
                fit = func(pop[i])
                if fit < pbest_fit[i]:
                    pbest[i] = pop[i]
                    pbest_fit[i] = fit
                    if fit < gbest_fit:
                        gbest = pop[i]
                        gbest_fit = fit

        return gbest