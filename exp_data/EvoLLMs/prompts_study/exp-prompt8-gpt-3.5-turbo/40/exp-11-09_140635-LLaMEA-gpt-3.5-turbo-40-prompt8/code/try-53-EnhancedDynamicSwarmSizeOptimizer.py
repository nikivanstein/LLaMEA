class EnhancedDynamicSwarmSizeOptimizer(EnhancedAdaptiveMutationRateOptimizer):
    def __init__(self, budget, dim, num_swarms=5, swarm_size=10, inertia_weight=0.5, cognitive_weight=1.5, social_weight=2.0, initial_mutation_rate=0.1, adaptive_mutation_rate=0.1):
        super().__init__(budget, dim, num_swarms, swarm_size, inertia_weight, cognitive_weight, social_weight, initial_mutation_rate, adaptive_mutation_rate)

    def __call__(self, func):
        swarms = [np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        velocities = [np.zeros((self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
        best_positions = [swarms[i][np.argmin([func(p) for p in swarms[i]])] for i in range(self.num_swarms)]
        global_best_position = best_positions[0].copy()
        
        for t in range(self.budget):
            if t % (self.budget // 5) == 0:  # Dynamic adjustment of swarm size
                self.swarm_size = max(5, self.swarm_size // 2)
                swarms = [np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
                velocities = [np.zeros((self.swarm_size, self.dim)) for _ in range(self.num_swarms)]
                best_positions = [swarms[i][np.argmin([func(p) for p in swarms[i]])] for i in range(self.num_swarms)]
                
            for i in range(self.num_swarms):
                for j in range(self.swarm_size):
                    cognitive_component = self.cognitive_weight * np.random.rand(self.dim) * (best_positions[i] - swarms[i][j])
                    social_component = self.social_weight * np.random.rand(self.dim) * (global_best_position - swarms[i][j])
                    adaptive_inertia = self.inertia_weight * (1 - t / self.budget)
                    velocities[i][j] = adaptive_inertia * velocities[i][j] + cognitive_component + social_component
                    
                    if np.random.rand() < self.adaptive_mutation_rate * (1 - t / self.budget):
                        opposite_position = 2 * np.mean(swarms[i]) - swarms[i][j]
                        swarms[i][j] = np.clip(opposite_position + np.random.normal(0, 1, self.dim), -5.0, 5.0)
                    else:
                        swarms[i][j] = np.clip(swarms[i][j] + velocities[i][j], -5.0, 5.0)
                    
                    if func(swarms[i][j]) < func(best_positions[i]):
                        best_positions[i] = swarms[i][j]
                    if func(swarms[i][j]) < func(global_best_position):
                        global_best_position = swarms[i][j]
                        self.cognitive_weight *= 0.9
                        self.social_weight *= 0.9

        return global_best_position