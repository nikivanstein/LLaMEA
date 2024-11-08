import numpy as np

class DE_PSO_Optimizer:
    def __init__(self, budget, dim, pop_size=50, max_iter=1000, F=0.5, CR=0.9, w=0.5, c1=2.0, c2=2.0):
        self.budget, self.dim, self.pop_size, self.max_iter, self.F, self.CR, self.w, self.c1, self.c2 = budget, dim, pop_size, max_iter, F, CR, w, c1, c2

    def __call__(self, func):
        def de_pso(func):
            population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
            pbest_pos = population.copy()
            pbest_val = np.array([func(ind) for ind in population])
            gbest_idx = np.argmin(pbest_val)
            gbest_pos = pbest_pos[gbest_idx].copy()
            velocities = np.zeros((self.pop_size, self.dim))
            
            for _ in range(self.max_iter):
                for i in range(self.pop_size):
                    r1, r2 = np.random.rand(), np.random.rand()
                    velocities[i] = self.w * velocities[i] + self.c1 * r1 * (pbest_pos[i] - population[i]) + self.c2 * r2 * (gbest_pos - population[i])
                    population[i] = np.clip(population[i] + velocities[i], -5.0, 5.0)
                    new_val = func(population[i])
                    
                    if new_val < pbest_val[i]:
                        pbest_val[i] = new_val
                        pbest_pos[i] = population[i]
                        if new_val < pbest_val[gbest_idx]:
                            gbest_idx = i
                            gbest_pos = pbest_pos[i]
                
                if func.calls >= self.budget:
                    break
            
            return gbest_pos
        
        return de_pso(func)