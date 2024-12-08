class MemoryEnhancedQIPSO(EnhancedQIPSO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.memory = np.zeros((self.num_particles, self.dim))

    def __call__(self, func):
        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                fitness = func(self.particles[i])
                if fitness < self.personal_best_values[i]:
                    self.personal_best_values[i] = fitness
                    self.personal_best_positions[i] = self.particles[i].copy()
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = self.particles[i].copy()
                
                r = np.random.uniform(0, 1, self.dim)
                dynamic_alpha = 0.9 - (_ / self.max_iter) * 0.5
                dynamic_beta = 0.4 + (_ / self.max_iter) * 0.6
                self.velocities[i] = dynamic_alpha * self.velocities[i] + dynamic_beta * (self.personal_best_positions[i] - self.particles[i]) + dynamic_beta * (self.global_best_position - self.particles[i]) + 0.1 * self.memory[i]
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i] * r, self.lower_bound, self.upper_bound)
                if fitness < self.personal_best_values[i]:
                    self.memory[i] = 0.5 * self.memory[i] + 0.5 * self.personal_best_positions[i]
        
        return self.global_best_value