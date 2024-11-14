class EnhancedAdaptiveStepHybridPSOLocalSearch(HybridPSOLocalSearch):
    def local_search(self, particle, func):
        best_particle = np.copy(particle)
        step_size = 0.1
        history = []
        for _ in range(5):
            new_particle = np.clip(best_particle + step_size * np.random.randn(self.dim), self.lb, self.ub)
            history.append(func(new_particle))
            if func(new_particle) < func(best_particle):
                best_particle = np.copy(new_particle)
                step_size *= 0.9  # Decrease step size if better solution found
            else:
                step_size *= 1.1  # Increase step size if no improvement
        return best_particle