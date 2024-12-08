import numpy as np

class EnhancedHybridPSOADE_AdaptiveInertia_Improved_Optimized_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso_ade_improved():
            best_pos = np.random.uniform(-5.0, 5.0, self.dim)
            best_val = np.inf
            inertia_weight = 0.5
            budget_half = self.budget // 2
            inertia_dec_factor = 0.9 / budget_half

            for _ in range(budget_half):
                new_pos = best_pos + np.random.uniform(-1, 1, self.dim) * (np.random.uniform(-5.0, 5.0, self.dim) - best_pos)
                new_val = func(new_pos)

                best_val, best_pos = (new_val, new_pos) if new_val < best_val else (best_val, best_pos)

                inertia_weight = max(0.4, inertia_weight * (1 - _ * inertia_dec_factor))

                r = np.random.choice(range(self.dim), 1)[0]
                mutant = best_pos + 0.5 * (best_pos - best_pos) + inertia_weight * (new_pos - best_pos)
                trial = np.where(np.random.uniform(size=self.dim) < 0.5, mutant, best_pos)

                trial_val = func(trial)
                best_val, best_pos = (trial_val, trial) if trial_val < best_val else (best_val, best_pos)

            return best_val

        return pso_ade_improved()