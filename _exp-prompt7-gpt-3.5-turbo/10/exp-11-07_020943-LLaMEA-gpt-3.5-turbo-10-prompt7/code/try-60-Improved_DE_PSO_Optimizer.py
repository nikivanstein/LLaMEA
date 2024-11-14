import numpy as np

class Improved_DE_PSO_Optimizer:
    def __init__(self, budget, dim, pop_size=50, max_iter=1000, F=0.5, CR=0.9, w=0.5, c1=2.0, c2=2.0):
        self.budget, self.dim, self.pop_size, self.max_iter, self.F, self.CR, self.w, self.c1, self.c2 = budget, dim, pop_size, max_iter, F, CR, w, c1, c2

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        pbest_pos = population.copy()
        pbest_val = np.array([func(ind) for ind in population])
        gbest_idx = np.argmin(pbest_val)
        gbest_pos = pbest_pos[gbest_idx].copy()
        velocities = np.zeros((self.pop_size, self.dim))

        for _ in range(self.max_iter):
            rand_values = np.random.rand(self.pop_size, 2)
            w, c1, c2 = self.w, self.c1, self.c2
            c1r1, c2r2 = c1 * rand_values[:, 0], c2 * rand_values[:, 1]
            velocities = w * velocities + c1r1[:, np.newaxis] * (pbest_pos - population) + c2r2[:, np.newaxis] * (gbest_pos - population)
            population = np.clip(population + velocities, -5.0, 5.0)
            new_vals = np.array([func(ind) for ind in population])

            updates = new_vals < pbest_val
            pbest_val[updates] = new_vals[updates]
            pbest_pos[updates] = population[updates]
            gbest_idx = np.argmin(pbest_val)
            gbest_pos = pbest_pos[gbest_idx]

            if _ >= self.budget:
                break

        return gbest_pos