import numpy as np

class FastQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.max_iter = int(budget / self.num_particles)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.alpha_min, self.alpha_max = 0.4, 0.9
        self.beta_min, self.beta_max = 0.2, 0.6
        self.inertia_min, self.inertia_max = 0.3, 0.9
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, self.dim))
        self.personal_best_positions = self.particles.copy()
        self.personal_best_values = np.full(self.num_particles, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_value = np.inf

    def __call__(self, func):
        for _ in range(self.max_iter):
            inertia_weight = self.inertia_min + ((_ + 1) / self.max_iter) * (self.inertia_max - self.inertia_min)
            for i in range(self.num_particles):
                fitness = func(self.particles[i])
                if fitness < self.personal_best_values[i]:
                    self.personal_best_values[i] = fitness
                    self.personal_best_positions[i] = self.particles[i].copy()
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = self.particles[i].copy()
                
                r = np.random.uniform(0, 1, self.dim)
                alpha = self.alpha_min + ((_ + 1) / self.max_iter) * (self.alpha_max - self.alpha_min)
                beta = self.beta_min + ((_ + 1) / self.max_iter) * (self.beta_max - self.beta_min)
                self.velocities[i] = inertia_weight * self.velocities[i] + alpha * (self.personal_best_positions[i] - self.particles[i]) + beta * (self.global_best_position - self.particles[i])
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i] * r, self.lower_bound, self.upper_bound)
        
        return self.global_best_value