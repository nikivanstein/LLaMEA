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

    def __call__(self, func):
        def de_pso(func):
            population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
            pbest_pos = population.copy()
            pbest_val = np.array([func(ind) for ind in population])
            gbest_idx = np.argmin(pbest_val)
            gbest_pos = pbest_pos[gbest_idx].copy()
            velocities = np.zeros((self.pop_size, self.dim))
            rand_values = np.random.rand(self.max_iter, self.pop_size, 2)

            for _ in range(self.budget):
                for i, (r1, r2) in zip(range(self.pop_size), rand_values[_]):
                    velocities[i] = self.w * velocities[i] + self.c1 * r1 * (pbest_pos[i] - population[i]) + self.c2 * r2 * (gbest_pos - population[i])
                    population[i] = np.clip(population[i] + velocities[i], -5.0, 5.0)
                    new_val = func(population[i])
                    if new_val < pbest_val[i]:
                        pbest_val[i], pbest_pos[i] = new_val, population[i].copy()
                        if new_val < pbest_val[gbest_idx]:
                            gbest_idx, gbest_pos = i, pbest_pos[i].copy()

            return gbest_pos

        return de_pso(func)