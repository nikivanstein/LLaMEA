import numpy as np

class HarmonySearchImprovedEfficient:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.lower_bound, self.upper_bound = budget, dim, -5.0, 5.0
        self.hm_size, self.par_min, self.par_max, self.bw_min, self.bw_max = 20, 0.4, 0.9, 0.01, 0.1
        self.pitch_adj_prob = np.random.rand(dim)

    def __call__(self, func):
        hm = np.random.uniform(self.lower_bound, self.upper_bound, (self.hm_size, self.dim))
        for _ in range(self.budget):
            harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            idx = np.random.randint(self.hm_size)
            mask = np.random.rand(self.dim) < self.pitch_adj_prob
            harmony = np.where(mask, harmony, hm[idx])
            harmony = np.clip(harmony + np.random.uniform(-0.1, 0.1, self.dim), self.lower_bound, self.upper_bound)
            hm = np.vstack((hm, harmony))
            hm = sorted(hm, key=lambda x: func(x))

        return hm[0]