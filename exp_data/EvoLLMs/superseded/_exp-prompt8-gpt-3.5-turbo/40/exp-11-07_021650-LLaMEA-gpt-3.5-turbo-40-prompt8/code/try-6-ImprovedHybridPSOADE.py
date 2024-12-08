import numpy as np

class ImprovedHybridPSOADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso_ade():
            best_pos = np.random.uniform(-5.0, 5.0, self.dim)
            best_val = np.inf
        
            for _ in range(self.budget):
                # PSO update
                new_pos = best_pos + np.random.uniform() * (best_pos - best_pos) + np.random.uniform(-5.0, 5.0, self.dim - 1) - best_pos
                new_val = func(new_pos)
        
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
        
                # ADE update
                for i in range(self.dim):
                    r = np.random.choice(range(self.dim), 3, replace=False)
                    mutant = best_pos + np.random.uniform() * (best_pos - best_pos) + np.random.uniform(-5.0, 5.0, self.dim) - best_pos
                    trial = np.where(np.random.rand(self.dim) < 0.5, mutant, best_pos)
                    
                    if func(trial) < func(best_pos):
                        best_pos = trial
        
            return best_val
        
        return pso_ade()