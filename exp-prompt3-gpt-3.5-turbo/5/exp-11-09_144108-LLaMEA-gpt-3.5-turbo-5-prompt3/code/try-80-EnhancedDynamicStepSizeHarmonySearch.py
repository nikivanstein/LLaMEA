import numpy as np

class EnhancedDynamicStepSizeHarmonySearch(DynamicStepSizeHarmonySearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = 0.1

    def __call__(self, func):
        harmonies = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.budget, self.dim))
        for _ in range(self.budget):
            new_harmony = harmonies[np.random.randint(0, self.budget)] + self.step_size * np.random.uniform(-0.1, 0.1) * (harmonies[np.random.randint(0, self.budget)] - harmonies[np.random.randint(0, self.budget)])
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)
            if func(new_harmony) < func(harmonies[-1]):
                harmonies[-1] = new_harmony
            # Opposite-based learning
            opposite_harmony = self.lower_bound + self.upper_bound - harmonies
            for idx, o_harm in enumerate(opposite_harmony):
                if func(o_harm) < func(harmonies[idx]):
                    harmonies[idx] = o_harm
            harmonies = harmonies[np.argsort([func(h) for h in harmonies])]

            self.step_size *= 0.995  # Dynamic step size adaptation based on individual harmony improvements

            # Introducing mutation for enhancing diversity
            if np.random.rand() < self.mutation_rate:
                random_idx = np.random.choice(self.budget)
                harmonies[random_idx] += np.random.normal(0, 0.1, size=self.dim)
                harmonies[random_idx] = np.clip(harmonies[random_idx], self.lower_bound, self.upper_bound)

        return harmonies[0]