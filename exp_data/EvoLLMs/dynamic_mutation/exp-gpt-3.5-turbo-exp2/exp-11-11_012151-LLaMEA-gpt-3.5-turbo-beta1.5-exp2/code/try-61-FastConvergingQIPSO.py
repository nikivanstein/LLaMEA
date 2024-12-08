import numpy as np

class FastConvergingQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.max_iter = int(budget / self.num_particles)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.alpha = 0.9
        self.beta = 0.4
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, self.dim))
        self.personal_best_positions = self.particles.copy()
        self.personal_best_values = np.full(self.num_particles, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_value = np.inf

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
                self.velocities[i] = dynamic_alpha * self.velocities[i] + dynamic_beta * (self.personal_best_positions[i] - self.particles[i]) + dynamic_beta * (self.global_best_position - self.particles[i])
                step_size = 0.1 * (1 - fitness / self.global_best_value)  # Variable step size based on fitness improvement
                self.particles[i] = np.clip(self.particles[i] + step_size * self.velocities[i] * r, self.lower_bound, self.upper_bound)
        
        return self.global_best_value