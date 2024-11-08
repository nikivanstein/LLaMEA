import numpy as np

class ImprovedHybridPSOADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso_ade():
            best_pos = np.random.uniform(-5.0, 5.0, self.dim)
            best_val = np.inf
            inertia_weight = 0.5  # Added inertia weight parameter

            for _ in range(self.budget):
                rand_uniform = np.random.uniform(size=self.dim)
                velocity = inertia_weight * rand_uniform * (best_pos - best_pos) + rand_uniform * (np.random.uniform(-5.0, 5.0, self.dim) - best_pos)  # Optimized velocity calculation
                new_pos = best_pos + velocity
                new_val = func(new_pos)
        
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
        
                r1, r2, r3 = np.random.choice(range(self.dim), 3, replace=False)
                mutant = best_pos + 0.5 * (best_pos - best_pos) + 0.5 * (np.random.uniform(-5.0, 5.0, self.dim) - best_pos)
                trial = np.where(np.random.uniform(size=self.dim) < 0.5, mutant, best_pos)
        
                if func(trial) < func(best_pos):
                    best_pos = trial
        
            return best_val
        
        return pso_ade()