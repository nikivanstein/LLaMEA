import numpy as np

class EnhancedImprovedPSOADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso_ade():
            best_pos = np.random.uniform(-5.0, 5.0, self.dim)
            best_val = np.inf
            inertia_weight = 0.5
            cognitive_weight = 0.5

            for _ in range(self.budget):
                # PSO update with inertia and cognitive components
                velocity = inertia_weight * (best_pos - best_pos) + cognitive_weight * np.random.uniform(size=self.dim) * (best_pos - best_pos) + cognitive_weight * np.random.uniform(size=self.dim) * (np.random.uniform(-5.0, 5.0, self.dim) - best_pos)
                new_pos = best_pos + velocity
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
        
            return best_val
        
        return pso_ade()