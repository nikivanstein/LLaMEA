import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, hms=10, hmcr=0.7, par=0.3, bw=0.01):
        self.budget, self.dim, self.hms, self.hmcr, self.par, self.bw, self.lower_bound, self.upper_bound = budget, dim, hms, hmcr, par, bw, -5.0, 5.0

    def generate_new_harmonies(self):
        harmonies = np.random.uniform(self.lower_bound, self.upper_bound, (self.hms, self.dim))
        mask = np.random.rand(self.hms, self.dim) < self.hmcr
        pitch_adjustments = np.random.uniform(-self.bw, self.bw, (self.hms, self.dim))
        
        harmonies += mask & (np.random.rand(self.hms, self.dim) < self.par) * pitch_adjustments
        
        return harmonies

    def __call__(self, func):
        harmonies = self.generate_new_harmonies()
        evaluations = 0

        while evaluations < self.budget:
            costs = np.array([func(h) for h in harmonies])
            evaluations += self.hms

        best_harmony = harmonies[np.argmin(costs)]
        return best_harmony