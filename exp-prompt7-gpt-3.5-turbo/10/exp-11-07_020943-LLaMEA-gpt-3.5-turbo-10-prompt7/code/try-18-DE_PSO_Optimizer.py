import numpy as np

class DE_PSO_Optimizer:
    def __init__(self, budget, dim, pop_size=50, max_iter=1000, F=0.5, CR=0.9, w=0.5, c1=2.0, c2=2.0):
        self.budget, self.dim, self.pop_size, self.max_iter, self.F, self.CR, self.w, self.c1, self.c2 = budget, dim, pop_size, max_iter, F, CR, w, c1, c2

    def __call__(self, func):
        def de_pso(func):
            population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
            pbest_pos, pbest_val = population.copy(), np.array([func(ind) for ind in population])
            gbest_idx = np.argmin(pbest_val)
            gbest_pos = pbest_pos[gbest_idx].copy()
            velocities = np.zeros((self.pop_size, self.dim))
            
            for _ in range(self.max_iter):
                r1, r2 = np.random.rand(self.pop_size, 1), np.random.rand(self.pop_size, 1)
                velocities = self.w * velocities + self.c1 * r1 * (pbest_pos - population) + self.c2 * r2 * (np.tile(gbest_pos, (self.pop_size, 1)) - population)
                population = np.clip(population + velocities, -5.0, 5.0)
                new_vals = np.apply_along_axis(func, 1, population)
                updates = new_vals < pbest_val
                pbest_pos[updates] = population[updates]
                pbest_val[updates] = new_vals[updates]
                gbest_idx = np.argmin(pbest_val)
                gbest_pos = pbest_pos[gbest_idx]
                if func.calls >= self.budget: break
            
            return gbest_pos

        return de_pso(func)