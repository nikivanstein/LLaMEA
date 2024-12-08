import numpy as np

class EnhancedQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 30
        self.max_iter = int(budget / self.num_particles)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.personal_best_positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.personal_best_values = np.full(self.num_particles, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_value = np.inf
        self.velocities = np.random.uniform(-0.1, 0.1, (self.num_particles, self.dim))
        self.alpha = 0.9
        self.beta = 0.4
        self.inertia_weights = np.ones(self.num_particles) * 0.5

    def __call__(self, func):
        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                fitness = func(self.personal_best_positions[i])
                if fitness < self.personal_best_values[i]:
                    self.personal_best_values[i] = fitness
                    self.personal_best_positions[i] = self.personal_best_positions[i].copy()
                    if fitness < self.global_best_value:
                        self.global_best_value = fitness
                        self.global_best_position = self.personal_best_positions[i].copy()
                
                r = np.random.uniform(0, 1, self.dim)
                self.velocities[i] = self.alpha * self.velocities[i] + self.beta * (self.personal_best_positions[i] - self.personal_best_positions[i]) + self.beta * (self.global_best_position - self.personal_best_positions[i])
                self.inertia_weights[i] = np.clip(0.5 + 0.5 * (self.personal_best_values[i] - fitness) / self.personal_best_values[i], 0.1, 1.0)
                self.personal_best_positions[i] = np.clip(self.personal_best_positions[i] + self.velocities[i] * r, self.lower_bound, self.upper_bound)
        
        return self.global_best_value