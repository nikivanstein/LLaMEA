import numpy as np

class ImprovedHybridPSOADE_Enhanced_Optimized_Improved_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso_ade():
            best_pos = np.random.uniform(-5.0, 5.0, self.dim)
            best_val = np.inf
            inertia_weight = 0.5

            for _ in range(self.budget):
                new_pos = best_pos + np.random.uniform(-1, 1, self.dim) * (np.random.uniform(-5.0, 5.0, self.dim) - best_pos)
                new_val = func(new_pos)
                
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                
                inertia_weight *= 0.9
                inertia_weight = max(0.4, min(inertia_weight, 0.9))

                mutant = best_pos + 0.5 * (best_pos - best_pos) + inertia_weight * (new_pos - best_pos)
                trial = np.where(np.random.uniform(size=self.dim) < 0.5, mutant, best_pos)

                if func(trial) < func(best_pos):
                    best_pos = trial

            return best_val
        
        return pso_ade()