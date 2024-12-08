class DynamicMutationEnhancedOppositionBasedDynamicInertiaWeightPSO(EnhancedOppositionBasedDynamicInertiaWeightPSO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.base_mutation_rate = 0.05  # Modified mutation rate
        self.mutation_rates = np.full((self.dim, self.dim), self.base_mutation_rate)
    
    def __call__(self, func):
        # Existing code
        for t in range(1, self.budget + 1):
            # Existing code
            
            # Dynamic Mutation Enhancement
            for i in range(self.dim):
                if fitness[i] < pbest_fitness[i]:  # Adapt mutation rate based on individual particle progress
                    self.mutation_rates[i] += 0.02  # Increase mutation rate for better performing particles
                else:
                    self.mutation_rates[i] -= 0.01  # Decrease mutation rate for worse performing particles
                self.mutation_rates[i] = np.clip(self.mutation_rates[i], 0.01, 0.1)  # Clip mutation rate within a range
                
                mutation_indices = np.random.choice(self.dim, int(self.dim * self.mutation_rates[i]), replace=False)
                swarm[i, mutation_indices] = np.random.uniform(-5.0, 5.0, len(mutation_indices))
            
            # Existing code