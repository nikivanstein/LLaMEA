import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim, hms=10, hmcr=0.7, par=0.3, bw=0.01):
        self.budget, self.dim, self.hms, self.hmcr, self.par, self.bw = budget, dim, hms, hmcr, par, bw
        self.lower_bound, self.upper_bound = -5.0, 5.0

    def generate_new_harmonies(self):
        harmonies = np.random.uniform(self.lower_bound, self.upper_bound, (self.hms, self.dim))
        for i in range(self.hms):
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    harmonies[i, j] += np.random.uniform(-self.bw, self.bw) * (np.random.rand() < self.par)
        return harmonies

    def __call__(self, func):
        harmonies = self.generate_new_harmonies()
        evaluations = 0

        while evaluations < self.budget:
            costs = np.array([func(h) for h in harmonies])
            evaluations += len(harmonies)

        best_harmony = harmonies[np.argmin(costs)]
        return best_harmony