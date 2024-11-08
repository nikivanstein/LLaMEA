import numpy as np

class ImprovedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.n_particles = 30
        self.c1 = 2.05
        self.c2 = 2.05
        self.inertia_weight = 0.9
        self.inertia_damping = 0.99
        self.max_velocity = (self.upper_bound - self.lower_bound) * 0.2
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.n_particles, dim))
        self.velocities = np.zeros((self.n_particles, dim))
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.full(self.n_particles, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf
    
    def __call__(self, func):
        func_calls = 0
        inertia_weight = self.inertia_weight  # Local copy for quicker access
        while func_calls < self.budget:
            fitness_values = np.apply_along_axis(func, 1, self.positions)
            func_calls += self.n_particles
            
            for i in range(self.n_particles):
                fitness = fitness_values[i]
                if fitness < self.pbest_scores[i]:
                    self.pbest_scores[i] = fitness
                    self.pbest_positions[i] = self.positions[i]
                if fitness < self.gbest_score:
                    self.gbest_score = fitness
                    self.gbest_position = self.positions[i]
            
            r1, r2 = np.random.rand(self.n_particles, self.dim), np.random.rand(self.n_particles, self.dim)
            cognitive_term = self.c1 * r1 * (self.pbest_positions - self.positions)
            social_term = self.c2 * r2 * (self.gbest_position - self.positions)
            self.velocities = inertia_weight * self.velocities + cognitive_term + social_term
            
            # Velocity clamping
            np.clip(self.velocities, -self.max_velocity, self.max_velocity, out=self.velocities)
            self.positions += self.velocities
            
            # Ensure position bounds
            np.clip(self.positions, self.lower_bound, self.upper_bound, out=self.positions)
            
            # Adaptive inertia weight adjustment
            inertia_weight *= self.inertia_damping