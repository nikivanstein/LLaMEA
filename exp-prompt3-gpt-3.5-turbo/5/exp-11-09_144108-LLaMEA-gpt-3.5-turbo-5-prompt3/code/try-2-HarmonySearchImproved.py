import numpy as np

class HarmonySearchImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.5  # Dynamically adjust upper bound to 5.5

    def __call__(self, func):
        harmonies = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.budget, self.dim))
        for _ in range(self.budget):
            new_harmony = np.clip(np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim), self.lower_bound, self.upper_bound)
            if func(new_harmony) < func(harmonies[-1]):
                harmonies[-1] = new_harmony
            harmonies = harmonies[np.argsort([func(h) for h in harmonies])]
        return harmonies[0]