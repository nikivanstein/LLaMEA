import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.n_particles = 30
        self.c1 = 2.05  # cognitive coefficient
        self.c2 = 2.05  # social coefficient
        self.inertia_weight = 0.9  # initial inertia weight
        self.inertia_damping = 0.99  # damping factor for inertia
        self.max_velocity = (self.upper_bound - self.lower_bound) * 0.2
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.n_particles, dim))
        self.velocities = np.zeros((self.n_particles, dim))
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.full(self.n_particles, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf
    
    def __call__(self, func):
        func_calls = 0
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

            r1, r2 = np.random.rand(2, self.n_particles, self.dim)
            cognitive_terms = self.c1 * r1 * (self.pbest_positions - self.positions)
            social_terms = self.c2 * r2 * (self.gbest_position - self.positions)
            self.velocities = self.inertia_weight * self.velocities + cognitive_terms + social_terms

            self.velocities = np.clip(self.velocities, -self.max_velocity, self.max_velocity)
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            self.inertia_weight *= self.inertia_damping