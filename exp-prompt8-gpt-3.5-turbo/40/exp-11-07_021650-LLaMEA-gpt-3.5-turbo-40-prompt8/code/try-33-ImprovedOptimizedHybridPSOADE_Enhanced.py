import numpy as np

class ImprovedOptimizedHybridPSOADE_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso_ade_optimized():
            best_pos = np.random.uniform(-5.0, 5.0, self.dim)
            best_val = np.inf
            inertia_weight = 0.5
            inertia_decay = 0.9
            crossover_prob = 0.5
            
            for _ in range(self.budget):
                rand_uniform = np.random.uniform(size=self.dim)
                new_pos = best_pos + rand_uniform * (best_pos - best_pos) + rand_uniform * (np.random.uniform(-5.0, 5.0, self.dim) - best_pos)
                new_val = func(new_pos)
                
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                
                inertia_weight = max(inertia_weight * inertia_decay, 0.4)
                
                r1, r2, r3 = np.random.choice(range(self.dim), 3, replace=False)
                mutant = best_pos + 0.5 * (best_pos - best_pos) + inertia_weight * (np.random.uniform(-5.0, 5.0, self.dim) - best_pos)
                trial = np.where(np.random.uniform(size=self.dim) < crossover_prob, mutant, best_pos)

                donor = np.random.choice(range(self.budget))
                mutant = best_pos + 0.5 * (best_pos - best_pos) + inertia_weight * (best_pos - best_pos) + inertia_weight * (np.random.uniform(-5.0, 5.0, self.dim) - best_pos) - np.random.uniform(0, 1, self.dim) * (best_pos - best_pos)
                trial = np.where(np.random.uniform(size=self.dim) < crossover_prob, mutant, best_pos)
                
                if func(trial) < func(best_pos):
                    best_pos = trial
            
            return best_val
        
        return pso_ade_optimized()