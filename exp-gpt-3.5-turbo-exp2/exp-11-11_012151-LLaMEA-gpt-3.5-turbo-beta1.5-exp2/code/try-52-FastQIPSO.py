import numpy as np

class FastQIPSO(QIPSO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.initial_inertia = 0.9
        self.final_inertia = 0.4

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
                inertia_weight = self.initial_inertia - (_ / self.max_iter) * (self.initial_inertia - self.final_inertia)
                self.velocities[i] = inertia_weight * self.velocities[i] + self.beta * (self.personal_best_positions[i] - self.particles[i]) + self.beta * (self.global_best_position - self.particles[i])
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i] * r, self.lower_bound, self.upper_bound)
        
        return self.global_best_value