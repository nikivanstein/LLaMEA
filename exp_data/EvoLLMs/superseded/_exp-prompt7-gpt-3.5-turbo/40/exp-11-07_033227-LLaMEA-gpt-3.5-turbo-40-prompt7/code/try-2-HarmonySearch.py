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

    def generate_new_harmony(self):
        harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:  # Memory consideration
                if np.random.rand() < self.par:  # Pitch adjustment
                    harmony[i] = harmony[i] + np.random.uniform(-self.bw, self.bw)
        return harmony

    def __call__(self, func):
        harmonies = [self.generate_new_harmony() for _ in range(self.hms)]
        evaluations = 0

        while evaluations < self.budget:
            for i in range(len(harmonies)):
                if evaluations >= self.budget:
                    break
                cost = func(harmonies[i])
                evaluations += 1

        best_harmony = min(harmonies, key=func)
        return best_harmony