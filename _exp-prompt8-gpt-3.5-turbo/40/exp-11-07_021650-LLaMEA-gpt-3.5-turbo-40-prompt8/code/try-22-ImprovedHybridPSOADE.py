import numpy as np

class ImprovedHybridPSOADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso_ade():
            best_pos = np.random.uniform(-5.0, 5.0, self.dim)
            best_val = np.inf
            w = 0.5  # Inertia weight
            
            for _ in range(self.budget):
                # PSO update with adaptive inertia weight
                rand_uniform = np.random.uniform(size=self.dim)
                new_pos = best_pos + w * (best_pos - best_pos) + rand_uniform * (np.random.uniform(-5.0, 5.0, self.dim) - best_pos)
                new_val = func(new_pos)
                
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    
                # ADE update with optimized mutation strategy
                r1, r2, r3 = np.random.choice(range(self.dim), 3, replace=False)
                mutant = best_pos + 0.5 * (best_pos - best_pos) + 0.5 * (np.random.uniform(-5.0, 5.0, self.dim) - best_pos)
                trial = np.where(np.random.uniform(size=self.dim) < 0.5, mutant, best_pos)
                
                if func(trial) < func(best_pos):
                    best_pos = trial
                    
                # Update inertia weight dynamically
                w = 0.5 + 0.5 * (self.budget - _) / self.budget  # Linearly decrease inertia weight
                
            return best_val
        
        return pso_ade()