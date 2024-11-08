import numpy as np

class Enhanced_DE_PSO_Optimizer:
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
            rand_values = np.random.rand(self.max_iter, self.pop_size, 2)
            
            for _ in range(self.max_iter):
                rand_values_curr = rand_values[_]
                velocities = self.w * velocities + self.c1 * rand_values_curr[:, 0, None] * (pbest_pos - population) + self.c2 * rand_values_curr[:, 1, None] * (gbest_pos - population)
                population = np.clip(population + velocities, -5.0, 5.0)
                new_vals = np.apply_along_axis(func, 1, population)
                
                updates = new_vals < pbest_val
                pbest_val[updates] = new_vals[updates]
                pbest_pos[updates] = population[updates]
                
                gbest_idx = np.argmin(pbest_val)
                gbest_pos = pbest_pos[gbest_idx]
                
                if _ >= self.budget:
                    break
            
            return gbest_pos
        
        return de_pso(func)