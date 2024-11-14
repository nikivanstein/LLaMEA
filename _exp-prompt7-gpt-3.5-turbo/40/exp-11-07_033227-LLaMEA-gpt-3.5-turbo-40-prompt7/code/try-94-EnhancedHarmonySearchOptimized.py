import numpy as np
import concurrent.futures

class EnhancedHarmonySearchOptimized:
    def __init__(self, budget, dim, hms=10, hmcr=0.7, par=0.3, bw=0.01):
        self.budget, self.dim, self.hms, self.hmcr, self.par, self.bw = budget, dim, hms, hmcr, par, bw
        self.lower_bound, self.upper_bound = -5.0, 5.0

    def generate_new_harmonies(self):
        harmonies = np.random.uniform(self.lower_bound, self.upper_bound, (self.hms, self.dim))
        mask = np.random.rand(self.hms, self.dim) < self.hmcr
        pitch_adjustments = np.random.uniform(-self.bw, self.bw, (self.hms, self.dim))
        
        harmonies = np.where(mask, harmonies + pitch_adjustments, harmonies)
        
        return harmonies

    def evaluate_func(self, func, harmonies):
        return np.array(list(map(func, harmonies)))

    def __call__(self, func):
        evaluations = 0
        harmonies = self.generate_new_harmonies()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            while evaluations < self.budget:
                costs = self.evaluate_func(func, harmonies)
                evaluations += len(harmonies)

        best_harmony = harmonies[np.argmin(costs)]
        return best_harmony