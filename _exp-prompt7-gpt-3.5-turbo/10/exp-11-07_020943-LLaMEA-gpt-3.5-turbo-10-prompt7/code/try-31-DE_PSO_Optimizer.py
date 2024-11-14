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
            pbest = np.stack((population, [func(ind) for ind in population]), axis=-1)
            gbest_idx = np.argmin(pbest[:, 1])
            gbest_pos = pbest[gbest_idx, 0].copy()
            velocities = np.zeros((self.pop_size, self.dim))
            rand_values = np.random.rand(self.max_iter, self.pop_size, 2)  # Pre-calculate random values
            
            for _ in range(self.max_iter):
                for i in range(self.pop_size):
                    r1, r2 = rand_values[_, i]
                    velocities[i] = self.w * velocities[i] + self.c1 * r1 * (pbest[i, 0] - population[i]) + self.c2 * r2 * (gbest_pos - population[i])
                    population[i] = np.clip(population[i] + velocities[i], -5.0, 5.0)
                    new_val = func(population[i])
                    
                    if new_val < pbest[i, 1]:
                        pbest[i] = [population[i], new_val]
                        if new_val < pbest[gbest_idx, 1]:
                            gbest_idx = i
                            gbest_pos = pbest[i, 0].copy()
                
                if _ >= self.budget:
                    break
            
            return gbest_pos
        
        return de_pso(func)