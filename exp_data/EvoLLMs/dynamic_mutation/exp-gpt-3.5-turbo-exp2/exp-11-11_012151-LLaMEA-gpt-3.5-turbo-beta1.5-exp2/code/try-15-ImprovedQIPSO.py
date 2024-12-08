class ImprovedQIPSO(QIPSO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.inertia_weights = np.linspace(0.9, 0.4, self.max_iter)  # Dynamic inertia weight adjustment

    def __call__(self, func):
        for t in range(self.max_iter):
            for i in range(self.num_particles):
                fitness = func(self.particles[i])
                if fitness < self.personal_best_values[i]:
                    self.personal_best_values[i] = fitness
                    self.personal_best_positions[i] = self.particles[i].copy()
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = self.particles[i].copy()
                
                r = np.random.uniform(0, 1, self.dim)
                self.velocities[i] = self.inertia_weights[t] * self.velocities[i] + self.beta * (self.personal_best_positions[i] - self.particles[i]) + self.beta * (self.global_best_position - self.particles[i])
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i] * r, self.lower_bound, self.upper_bound)
        
        return self.global_best_value