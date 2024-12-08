import numpy as np

class DE_PSO_Optimizer:
    def __init__(self, budget, dim, pop_size=50, max_iter=1000, F=0.5, CR=0.9, w=0.5, c1=2.0, c2=2.0):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F
        self.CR = CR
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.velocities = np.zeros((pop_size, dim))
        self.pbest_pos = np.zeros((pop_size, dim))

    def __call__(self, func):
        def de_pso(func):
            population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
            np.copyto(self.pbest_pos, population)
            pbest_val = np.array([func(ind) for ind in population])
            gbest_idx = np.argmin(pbest_val)
            gbest_pos = self.pbest_pos[gbest_idx].copy()
            rand_values = np.random.rand(self.max_iter, self.pop_size, 2)  # Pre-calculate random values

            for _ in range(self.max_iter):
                for i in range(self.pop_size):
                    r1, r2 = rand_values[_, i]
                    self.velocities[i] = self.w * self.velocities[i] + self.c1 * r1 * (self.pbest_pos[i] - population[i]) + self.c2 * r2 * (gbest_pos - population[i])
                    population[i] = np.clip(population[i] + self.velocities[i], -5.0, 5.0)
                    new_val = func(population[i])

                    if new_val < pbest_val[i]:
                        pbest_val[i] = new_val
                        np.copyto(self.pbest_pos[i], population[i])
                        if new_val < pbest_val[gbest_idx]:
                            gbest_idx = i
                            gbest_pos = self.pbest_pos[i].copy()

                if _ >= self.budget:
                    break

            return gbest_pos

        return de_pso(func)