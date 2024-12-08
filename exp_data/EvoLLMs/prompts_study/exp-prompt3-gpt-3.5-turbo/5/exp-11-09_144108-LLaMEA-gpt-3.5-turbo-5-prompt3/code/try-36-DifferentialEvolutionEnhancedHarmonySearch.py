import numpy as np

class DifferentialEvolutionEnhancedHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

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
            # Differential Evolution Strategy
            for idx in range(self.budget):
                candidate = harmonies[np.random.choice(np.setdiff1d(np.arange(self.budget), idx), 3, replace=False)]
                mutant = candidate[0] + 0.5 * (candidate[1] - candidate[2])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                if func(mutant) < func(harmonies[idx]):
                    harmonies[idx] = mutant
            harmonies = harmonies[np.argsort([func(h) for h in harmonies])]
        return harmonies[0]