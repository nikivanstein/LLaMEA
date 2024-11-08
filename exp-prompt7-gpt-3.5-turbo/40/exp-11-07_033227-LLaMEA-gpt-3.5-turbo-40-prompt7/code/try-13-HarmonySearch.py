import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, hms=10, hmcr=0.7, par=0.3, bw=0.01):
        self.budget = budget
        self.dim = dim
        self.hms = hms
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def generate_new_harmonies(self):
        harmonies = np.random.uniform(self.lower_bound, self.upper_bound, (self.hms, self.dim))
        mask_hmcr = np.random.rand(self.hms, self.dim) < self.hmcr
        mask_par = np.random.rand(self.hms, self.dim) < self.par
        pitch_adjustments = np.random.uniform(-self.bw, self.bw, (self.hms, self.dim))
        
        harmonies = np.where(mask_hmcr & mask_par, harmonies + pitch_adjustments, harmonies)
        
        return harmonies

    def __call__(self, func):
        harmonies = self.generate_new_harmonies()
        costs = np.array([func(h) for h in harmonies])
        evaluations = len(harmonies)

        while evaluations < self.budget:
            new_harmonies = self.generate_new_harmonies()
            new_costs = np.array([func(h) for h in new_harmonies])
            evaluations += len(new_harmonies)
            harmonies = np.where(new_costs < costs, new_harmonies, harmonies)
            costs = np.where(new_costs < costs, new_costs, costs)

        best_harmony = harmonies[np.argmin(costs)]
        return best_harmony