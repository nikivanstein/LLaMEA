import numpy as np

class ImprovedEnhancedHybridPSOADE_AdaptiveInertia_Improved_Optimized_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso_ade_improved():
            best_pos = np.random.uniform(-5.0, 5.0, self.dim)
            best_val = np.inf
            inertia_weight = 0.5
            half_budget = self.budget // 2

            for _ in range(half_budget):
                new_pos = best_pos + np.random.uniform(-1, 1, self.dim) * (np.random.uniform(-5.0, 5.0, self.dim) - best_pos)
                new_val = func(new_pos)

                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos

                inertia_weight = max(0.4, 0.9 * inertia_weight * (1 - _ / half_budget))

                r = np.random.choice(range(self.dim), 1)[0]
                mutant = best_pos + 0.5 * (best_pos - best_pos) + inertia_weight * (new_pos - best_pos)
                trial = np.where(np.random.uniform(size=self.dim) < 0.5, mutant, best_pos)

                trial_val = func(trial)
                if trial_val < best_val:
                    best_pos = trial
                    best_val = trial_val

            return best_val

        return pso_ade_improved()