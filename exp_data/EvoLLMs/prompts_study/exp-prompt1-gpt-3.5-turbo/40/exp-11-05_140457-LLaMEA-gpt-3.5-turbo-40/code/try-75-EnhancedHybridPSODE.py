class EnhancedHybridPSODE(HybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_radius = 0.1
        self.local_search_prob = 0.1
        
    def __call__(self, func):
        for t in range(self.budget):
            func_val = func(self.global_best)

            # Dynamically adjust local search radius based on function landscape
            self.local_search_radius = 0.1 + 0.05 * np.tanh(func_val)

            if np.random.rand() < self.local_search_prob:  # Local Search
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
            
            else:  # Original HybridPSODE update
                super().__call__(func)
        return self.global_best