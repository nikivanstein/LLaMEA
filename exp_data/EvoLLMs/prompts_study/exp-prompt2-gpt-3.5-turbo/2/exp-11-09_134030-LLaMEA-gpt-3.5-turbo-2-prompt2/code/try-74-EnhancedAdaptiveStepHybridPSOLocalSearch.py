class EnhancedAdaptiveStepHybridPSOLocalSearch(HybridPSOLocalSearch):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.memory = np.zeros(dim)  # Initialize memory for each dimension
        self.neighborhood_size = 3

    def local_search(self, particle, func):
        best_particle = np.copy(particle)
        step_size = 0.1
        for _ in range(5):
            for _ in range(self.neighborhood_size):
                new_particle = np.clip(best_particle + step_size * np.random.randn(self.dim) + 0.2 * self.memory, self.lb, self.ub)
                current_val = func(best_particle)
                new_val = func(new_particle)
                if new_val < current_val:
                    best_particle = np.copy(new_particle)
                    step_size *= 0.9  # Decrease step size if better solution found
                    self.memory = 0.9 * self.memory + 0.1 * (best_particle - particle)  # Updated memory ratio
                else:
                    step_size *= 1.1  # Increase step size if no improvement
            # Adaptive mutation strategy based on exploration-exploitation balance
            mutation_rate = 0.1 + 0.1 * np.tanh(0.01 * np.linalg.norm(best_particle - particle))
            best_particle += mutation_rate * np.random.randn(self.dim)
            # Adaptive neighborhood size adjustment based on function landscape curvature
            self.neighborhood_size = min(5, max(2, int(np.abs(np.mean(func(best_particle) - func(particle)))))
        return best_particle