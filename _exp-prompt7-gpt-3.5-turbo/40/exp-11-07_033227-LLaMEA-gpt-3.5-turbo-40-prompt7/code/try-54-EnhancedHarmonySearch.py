import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim, hms=10, hmcr=0.7, par=0.3, bw=0.01):
        self.budget, self.dim, self.hms, self.hmcr, self.par, self.bw = budget, dim, hms, hmcr, par, bw
        self.lower_bound, self.upper_bound = -5.0, 5.0

    def generate_new_harmonies(self):
        harmonies = np.random.uniform(self.lower_bound, self.upper_bound, (self.hms, self.dim))
        mask_hmcr, mask_par = np.random.rand(self.hms, self.dim) < self.hmcr, np.random.rand(self.hms, self.dim) < self.par
        
        harmonies += np.where(mask_hmcr & mask_par, np.random.uniform(-self.bw, self.bw, (self.hms, self.dim)), 0)
        
        return harmonies

    def __call__(self, func):
        harmonies = self.generate_new_harmonies()
        costs = np.array([func(h) for h in harmonies])

        return harmonies[np.argmin(costs)]