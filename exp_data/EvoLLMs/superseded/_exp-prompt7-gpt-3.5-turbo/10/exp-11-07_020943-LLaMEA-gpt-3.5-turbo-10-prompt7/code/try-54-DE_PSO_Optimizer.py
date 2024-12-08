import numpy as np

class DE_PSO_Optimizer:
    def __init__(self, budget, dim, pop_size=50, max_iter=1000, F=0.5, CR=0.9, w=0.5, c1=2.0, c2=2.0):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = min(budget, max_iter)  # Adjusted max_iter to be at most the budget
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
            rand_values = np.random.rand(self.max_iter, self.pop_size, 2)  # Pre-calculate random values
            
            for _ in range(self.max_iter):
                r_values = rand_values[_]
                velocities = self.w * velocities + self.c1 * r_values[:, 0, None] * (pbest_pos - population) + self.c2 * r_values[:, 1, None] * (gbest_pos - population)
                population = np.clip(population + velocities, -5.0, 5.0)
                new_vals = np.apply_along_axis(func, 1, population)
                
                update_mask = new_vals < pbest_val
                pbest_val[update_mask] = new_vals[update_mask]
                pbest_pos[update_mask] = population[update_mask]
                
                new_best_idx = np.argmin(new_vals)
                if new_vals[new_best_idx] < pbest_val[gbest_idx]:
                    gbest_idx = new_best_idx
                    gbest_pos = population[new_best_idx].copy()
                
                if _ >= self.budget:
                    break
            
            return gbest_pos
        
        return de_pso(func)