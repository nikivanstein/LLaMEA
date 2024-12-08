import numpy as np

class GWO_CS_Hybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.max_iter = budget // self.pop_size

    def __call__(self, func):
        lb = -5.0
        ub = 5.0
        pop = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        pbest = pop.copy()
        pbest_fit = np.array([func(ind) for ind in pbest])
        gbest = pbest[pbest_fit.argmin()]
        gbest_fit = pbest_fit.min()

        for _ in range(self.max_iter):
            # Gray Wolf Optimization (GWO) phase
            a = 2 - 2 * (_ / self.max_iter)
            for i in range(self.pop_size):
                A1, A2, A3 = 2 * a * np.random.rand() - a, 2 * a * np.random.rand() - a, 2 * a * np.random.rand() - a
                C1, C2, C3 = 2 * np.random.rand(), 2 * np.random.rand(), 2 * np.random.rand()
                D_alpha = np.abs(C1 * gbest - pop[i])
                D_beta = np.abs(C2 * pbest[i] - pop[i])
                D_delta = np.abs(C3 * np.mean(pbest) - pop[i])
                
                X1 = gbest - A1 * D_alpha
                X2 = pbest[i] - A2 * D_beta
                X3 = np.mean(pbest) - A3 * D_delta
                
                pop[i] = (X1 + X2 + X3) / 3
                pop[i] = np.clip(pop[i], lb, ub)
                fit = func(pop[i])
                
                if fit < pbest_fit[i]:
                    pbest[i] = pop[i]
                    pbest_fit[i] = fit
                    if fit < gbest_fit:
                        gbest = pop[i]
                        gbest_fit = fit

            # Cuckoo Search (CS) phase
            for i in range(self.pop_size):
                beta = 1.5
                L = np.random.normal(0, 1)
                X_new = pop[i] + L * beta * (pop[np.random.randint(0, self.pop_size)] - pop[np.random.randint(0, self.pop_size)])
                X_new = np.clip(X_new, lb, ub)
                fit = func(X_new)
                
                if fit < pbest_fit[i]:
                    pbest[i] = X_new
                    pbest_fit[i] = fit
                    if fit < gbest_fit:
                        gbest = X_new
                        gbest_fit = fit

        return gbest