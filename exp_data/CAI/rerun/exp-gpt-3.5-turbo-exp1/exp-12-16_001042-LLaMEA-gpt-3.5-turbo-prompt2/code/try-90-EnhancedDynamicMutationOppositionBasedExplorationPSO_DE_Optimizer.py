class EnhancedDynamicMutationOppositionBasedExplorationPSO_DE_Optimizer(DynamicMutationOppositionBasedExplorationPSO_DE_Optimizer):
    def __call__(self, func):
        def chaos_search(x):
            chaotic_seq = np.empty(self.dim)
            chaotic_seq[0] = x
            for i in range(1, self.dim):
                chaotic_seq[i] = 3.9 * chaotic_seq[i - 1] * (1 - chaotic_seq[i - 1])  # Logistic map
            return chaotic_seq
        
        for _ in range(int(self.budget * 0.038)):
            population = np.array([chaos_search(p) for p in population])
        
        for _ in range(int(self.budget * 0.962)):
            r1 = np.random.rand(self.swarm_size, self.dim)
            r2 = np.random.rand(self.swarm_size, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population)
            population += velocities
            ...
        return gbest