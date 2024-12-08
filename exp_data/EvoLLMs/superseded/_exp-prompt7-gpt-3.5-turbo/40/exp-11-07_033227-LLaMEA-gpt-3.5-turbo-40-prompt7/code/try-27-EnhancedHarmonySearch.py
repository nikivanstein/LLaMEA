import numpy as np

class EnhancedHarmonySearch:
    def __init__(self, budget, dim, hms=10, hmcr=0.7, par=0.3, bw=0.01):
        self.budget, self.dim, self.hms, self.hmcr, self.par, self.bw, self.lb, self.ub = budget, dim, hms, hmcr, par, bw, -5.0, 5.0

    def generate_new_harmonies(self):
        harmonies = np.random.uniform(self.lb, self.ub, (self.hms, self.dim))
        mask_hmcr, mask_par, pitch_adjustments = np.random.rand(self.hms, self.dim) < self.hmcr, np.random.rand(self.hms, self.dim) < self.par, np.random.uniform(-self.bw, self.bw, (self.hms, self.dim))
        harmonies = np.where(mask_hmcr & mask_par, harmonies + pitch_adjustments, harmonies)
        return harmonies

    def __call__(self, func):
        harmonies, evaluations = self.generate_new_harmonies(), 0

        while evaluations < self.budget:
            costs = np.array([func(h) for h in harmonies])
            evaluations += self.hms

        best_harmony = harmonies[np.argmin(costs)]
        return best_harmony