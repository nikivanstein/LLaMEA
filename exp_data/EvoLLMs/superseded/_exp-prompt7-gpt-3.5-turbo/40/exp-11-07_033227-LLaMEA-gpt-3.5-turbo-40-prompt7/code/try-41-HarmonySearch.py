import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, hms=10, hmcr=0.7, par=0.3, bw=0.01):
        self.budget, self.dim, self.hms, self.hmcr, self.par, self.bw = budget, dim, hms, hmcr, par, bw
        self.lower_bound, self.upper_bound = -5.0, 5.0
        self.masks_hmcr = np.random.rand(hms, dim) < hmcr
        self.masks_par = np.random.rand(hms, dim) < par
        self.pitch_adjustments = np.random.uniform(-bw, bw, (hms, dim))

    def generate_new_harmonies(self):
        harmonies = np.random.uniform(self.lower_bound, self.upper_bound, (self.hms, self.dim))
        harmonies = np.where(self.masks_hmcr & self.masks_par, harmonies + self.pitch_adjustments, harmonies)
        return harmonies

    def __call__(self, func):
        harmonies = self.generate_new_harmonies()
        costs = np.array([func(h) for h in harmonies])
        best_harmony = harmonies[np.argmin(costs)]
        return best_harmony