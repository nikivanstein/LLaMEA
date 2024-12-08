import numpy as np

class DynamicEnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.explore_prob = 0.5  # Initial exploration probability

    def __call__(self, func):
        harmonies = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.budget, self.dim))
        for _ in range(self.budget):
            new_harmony = harmonies[np.random.randint(0, self.budget)] + np.random.uniform(-0.1, 0.1) * (harmonies[np.random.randint(0, self.budget)] - harmonies[np.random.randint(0, self.budget)])
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            if func(new_harmony) < func(harmonies[-1]):
                harmonies[-1] = new_harmony
            # Opposite-based learning
            opposite_harmony = self.lower_bound + self.upper_bound - harmonies
            for idx, o_harm in enumerate(opposite_harmony):
                if func(o_harm) < func(harmonies[idx]):
                    harmonies[idx] = o_harm
            harmonies = harmonies[np.argsort([func(h) for h in harmonies])]
            # Adjust exploration probability dynamically
            self.explore_prob = 1.0 - (_ / self.budget)  # Linearly decrease exploration
        return harmonies[0]