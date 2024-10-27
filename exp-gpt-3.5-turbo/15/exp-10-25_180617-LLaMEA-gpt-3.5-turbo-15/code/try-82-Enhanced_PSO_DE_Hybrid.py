import numpy as np

class Enhanced_PSO_DE_Hybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.max_iter = budget // self.pop_size
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_min = 0.4
        self.w_max = 0.9
        self.mutation_rate = 0.3

    def __call__(self, func):
        lb = -5.0
        ub = 5.0
        pop = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        vel = np.zeros((self.pop_size, self.dim))
        pbest = pop.copy()
        pbest_fit = np.array([func(ind) for ind in pbest])
        gbest = pbest[pbest_fit.argmin()]
        gbest_fit = pbest_fit.min()
        inertia_weight = self.w_max

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                vel[i] = inertia_weight * vel[i] + self.c1 * r1 * (pbest[i] - pop[i]) + self.c2 * r2 * (gbest - pop[i])
                pop[i] = np.clip(pop[i] + vel[i], lb, ub)
                fit = func(pop[i])
                if fit < pbest_fit[i]:
                    pbest[i] = pop[i]
                    pbest_fit[i] = fit
                    if fit < gbest_fit:
                        gbest = pop[i]
                        gbest_fit = fit
                if np.random.rand() < self.mutation_rate:
                    pop[i] += np.random.uniform(-0.2, 0.2, self.dim)
                inertia_weight = self.w_max - (_ / self.max_iter) * (self.w_max - self.w_min)
        
        return gbest