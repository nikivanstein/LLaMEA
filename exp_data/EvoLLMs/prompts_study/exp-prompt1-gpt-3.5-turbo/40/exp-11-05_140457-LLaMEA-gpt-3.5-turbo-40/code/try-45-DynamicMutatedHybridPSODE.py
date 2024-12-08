class DynamicMutatedHybridPSODE(ImprovedHybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_prob = 0.3
    
    def __call__(self, func):
        for t in range(self.budget):
            if np.random.rand() < self.mutation_prob:  # Dynamic Mutation
                for i in range(self.population_size):
                    mutation_scale = np.random.uniform(0.01, 0.1)
                    perturbation = np.random.uniform(-mutation_scale, mutation_scale, self.dim)
                    candidate = np.clip(self.particles[i] + perturbation, -5.0, 5.0)
                    if func(candidate) < func(self.particles[i]):
                        self.particles[i] = candidate.copy()
            
            else:  # Original HybridPSODE update
                super().__call__(func)
        return self.global_best