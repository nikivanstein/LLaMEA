import numpy as np

class EnhancedAdaptiveStepHybridPSOLocalSearch(HybridPSOLocalSearch):
    def local_search(self, particle, func):
        best_particle = np.copy(particle)
        step_size = 0.1
        inertia_weight = 0.5
        for _ in range(5):
            new_particle = np.clip(best_particle + step_size * np.random.randn(self.dim), self.lb, self.ub)
            if func(new_particle) < func(best_particle):
                best_particle = np.copy(new_particle)
                step_size *= 0.9  # Decrease step size if better solution found
                inertia_weight = max(0.1, inertia_weight * 0.9)  # Adjust inertia weight for better exploitation
            else:
                step_size *= 1.1  # Increase step size if no improvement
                inertia_weight = min(0.9, inertia_weight * 1.1)  # Adjust inertia weight for better exploration
        return best_particle