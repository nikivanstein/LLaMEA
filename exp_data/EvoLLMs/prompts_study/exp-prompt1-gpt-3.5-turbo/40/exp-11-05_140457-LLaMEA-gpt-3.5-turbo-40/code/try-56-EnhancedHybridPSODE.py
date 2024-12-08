class EnhancedHybridPSODE(HybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.local_search_prob = 0.1
        self.local_search_radius = 0.1
        self.mutation_rate = 0.5
        
    def __call__(self, func):
        for t in range(self.budget):
            if np.random.rand() < self.local_search_prob:  # Local Search
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
            
            else:  # Updated Global Search with Dynamic Mutation
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim) * np.random.rand(self.dim) * self.mutation_rate
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
        return self.global_best