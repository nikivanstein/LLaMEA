class EnhancedQIPSO(QIPSO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.initial_alpha = self.alpha
        self.initial_beta = self.beta
        self.inertia_weight = 0.5

    def __call__(self, func):
        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                fitness = func(self.particles[i])
                if fitness < self.personal_best_values[i]:
                    self.personal_best_values[i] = fitness
                    self.personal_best_positions[i] = self.particles[i].copy()
                    self.alpha = self.initial_alpha + 0.1 * (self.global_best_value - fitness)
                    self.beta = self.initial_beta + 0.05 * (self.global_best_value - fitness)
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = self.particles[i].copy()
                
                r = np.random.uniform(0, 1, self.dim)
                self.velocities[i] = self.alpha * self.velocities[i] + self.beta * (self.personal_best_positions[i] - self.particles[i]) + self.beta * (self.global_best_position - self.particles[i])
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i] * r, self.lower_bound, self.upper_bound)
        
        return self.global_best_value