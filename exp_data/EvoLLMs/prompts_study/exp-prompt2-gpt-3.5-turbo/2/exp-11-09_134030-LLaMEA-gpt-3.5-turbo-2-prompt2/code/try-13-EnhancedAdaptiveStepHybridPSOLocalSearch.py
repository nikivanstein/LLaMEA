import numpy as np

class EnhancedAdaptiveStepHybridPSOLocalSearch(HybridPSOLocalSearch):
    def local_search(self, particle, func):
        best_particle = np.copy(particle)
        step_size = 0.1
        last_improvement = 0
        for _ in range(5):
            new_particle = np.clip(best_particle + step_size * np.random.randn(self.dim), self.lb, self.ub)
            if func(new_particle) < func(best_particle):
                best_particle = np.copy(new_particle)
                step_size *= 0.9  # Decrease step size if better solution found
                last_improvement = _
            else:
                if last_improvement != _ - 1:
                    step_size *= 1.1  # Increase step size if no improvement consecutively
        return best_particle