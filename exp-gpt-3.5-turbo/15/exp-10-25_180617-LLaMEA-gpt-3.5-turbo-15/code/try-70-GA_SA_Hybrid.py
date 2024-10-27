import numpy as np

class GA_SA_Hybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.max_iter = budget // self.pop_size
        self.mutation_rate = 0.1
        self.temperature = 1.0
        self.alpha = 0.9

    def __call__(self, func):
        lb = -5.0
        ub = 5.0
        pop = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        pbest = pop.copy()
        pbest_fit = np.array([func(ind) for ind in pbest])
        gbest = pbest[pbest_fit.argmin()]
        gbest_fit = pbest_fit.min()

        for _ in range(self.max_iter):
            new_pop = np.empty_like(pop)
            for i, ind in enumerate(pop):
                if np.random.rand() < self.mutation_rate:
                    new_ind = ind + np.random.normal(0, 1, self.dim)
                    new_ind = np.clip(new_ind, lb, ub)
                else:
                    new_ind = np.copy(ind)
                
                new_fit = func(new_ind)
                if new_fit < pbest_fit[i]:
                    pbest[i] = new_ind
                    pbest_fit[i] = new_fit
                    if new_fit < gbest_fit:
                        gbest = new_ind
                        gbest_fit = new_fit
                
                if np.random.rand() < np.exp((pbest_fit[i] - new_fit) / self.temperature):
                    new_pop[i] = new_ind
                else:
                    new_pop[i] = ind

            pop = np.copy(new_pop)
            self.temperature *= self.alpha

        return gbest