import numpy as np

class DynamicHarmonySearch(HarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        harmonies = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.budget, self.dim))
        pitch_adjust_rate = 0.01
        for _ in range(self.budget):
            new_harmony = np.clip(np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim) + np.random.uniform(-pitch_adjust_rate, pitch_adjust_rate, size=self.dim), self.lower_bound, self.upper_bound)
            if func(new_harmony) < func(harmonies[-1]):
                harmonies[-1] = new_harmony
            harmonies = harmonies[np.argsort([func(h) for h in harmonies])]
        return harmonies[0]