import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, hms=10, hmcr=0.7, par=0.3, bw=0.01):
        self.budget, self.dim, self.hms, self.hmcr, self.par, self.bw = budget, dim, hms, hmcr, par, bw
        self.lower_bound, self.upper_bound = -5.0, 5.0

    def generate_new_harmonies(self):
        self.harmonies = np.random.uniform(self.lower_bound, self.upper_bound, (self.hms, self.dim))
        self.mask_hmcr = np.random.rand(self.hms, self.dim) < self.hmcr
        self.mask_par = np.random.rand(self.hms, self.dim) < self.par
        self.pitch_adjustments = np.random.uniform(-self.bw, self.bw, (self.hms, self.dim))
        
        self.harmonies = np.where(self.mask_hmcr & self.mask_par, self.harmonies + self.pitch_adjustments, self.harmonies)
        
    def __call__(self, func):
        self.generate_new_harmonies()
        evaluations = 0

        while evaluations < self.budget:
            costs = np.array([func(h) for h in self.harmonies])
            evaluations += len(self.harmonies)

        best_harmony = self.harmonies[np.argmin(costs)]
        return best_harmony