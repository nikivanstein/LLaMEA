import numpy as np

class DynamicHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.step_size = 0.1

    def __call__(self, func):
        harmonies = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.budget, self.dim))
        for _ in range(self.budget):
            new_harmony = harmonies[np.random.randint(0, self.budget)] + np.random.uniform(-self.step_size, self.step_size) * (harmonies[np.random.randint(0, self.budget)] - harmonies[np.random.randint(0, self.budget)])
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            if func(new_harmony) < func(harmonies[-1]):
                harmonies[-1] = new_harmony
            # Opposite-based learning
            opposite_harmony = self.lower_bound + self.upper_bound - harmonies
            for idx, o_harm in enumerate(opposite_harmony):
                if func(o_harm) < func(harmonies[idx]):
                    harmonies[idx] = o_harm
            harmonies = harmonies[np.argsort([func(h) for h in harmonies])]
            # Adjust step size based on landscape
            best_harmony = harmonies[0]
            worst_harmony = harmonies[-1]
            self.step_size = np.linalg.norm(best_harmony - worst_harmony) / self.budget
        return harmonies[0]