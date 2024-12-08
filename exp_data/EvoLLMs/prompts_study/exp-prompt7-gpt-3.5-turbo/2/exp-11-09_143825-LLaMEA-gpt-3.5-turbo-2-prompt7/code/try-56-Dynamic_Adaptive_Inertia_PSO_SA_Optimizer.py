class Dynamic_Adaptive_Inertia_PSO_SA_Optimizer(Dynamic_Inertia_PSO_SA_Optimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.inertia_weights = np.full(budget, 0.7)
    
    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                fitness = func(self.particle_pos[i])
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best = self.particle_pos[i]
                if fitness < func(self.global_best):
                    self.global_best = self.particle_pos[i]

                performance_ratio = fitness / self.global_best_score
                dynamic_inertia_weight = 0.5 + 0.2 * performance_ratio  # Adaptive inertia weight adjustment
                self.inertia_weights[i] = dynamic_inertia_weight

                mutation_rate = 1.5  # Increased mutation rate for exploration
                new_vel = self.inertia_weights[i] * self.particle_vel[i] + mutation_rate * np.random.uniform(0, 1) * (self.global_best - self.particle_pos[i]) + mutation_rate * np.random.uniform(0, 1) * (self.particle_pos[i] - self.particle_pos[i])
                new_pos = self.particle_pos[i] + new_vel
                new_pos = np.clip(new_pos, -5.0, 5.0)
                new_fitness = func(new_pos)

                if new_fitness < fitness or np.random.rand() < np.exp((fitness - new_fitness) / self.temperature):
                    self.particle_pos[i] = new_pos
                    self.particle_vel[i] = new_vel

            self.temperature *= 0.95  # Annealing schedule

        return self.global_best