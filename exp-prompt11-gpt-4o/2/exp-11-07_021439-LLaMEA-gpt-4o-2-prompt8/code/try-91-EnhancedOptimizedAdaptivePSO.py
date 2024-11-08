import numpy as np

class EnhancedOptimizedAdaptivePSO:
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
        self.rng = np.random.default_rng()

    def __call__(self, func):
        func_calls = 0
        inertia_weight = self.inertia_weight  # Local caching for performance
        while func_calls < self.budget:
            batch_size = min(self.n_particles, self.budget - func_calls)
            fitness_values = np.apply_along_axis(func, 1, self.positions[:batch_size])
            func_calls += batch_size
            
            improved = fitness_values < self.pbest_scores[:batch_size]
            self.pbest_scores[:batch_size][improved] = fitness_values[improved]
            self.pbest_positions[:batch_size][improved] = self.positions[:batch_size][improved]
            
            min_index = np.argmin(self.pbest_scores[:batch_size])
            if self.pbest_scores[min_index] < self.gbest_score:
                self.gbest_score = self.pbest_scores[min_index]
                self.gbest_position = self.pbest_positions[min_index]
            
            r1, r2 = self.rng.random((2, batch_size, self.dim))  # Reduce redundant RNG calls
            cognitive_term = self.c1 * r1 * (self.pbest_positions[:batch_size] - self.positions[:batch_size])
            social_term = self.c2 * r2 * (self.gbest_position - self.positions[:batch_size])
            self.velocities[:batch_size] = np.clip(inertia_weight * self.velocities[:batch_size] + cognitive_term + social_term, -self.max_velocity, self.max_velocity)
            self.positions[:batch_size] = np.clip(self.positions[:batch_size] + self.velocities[:batch_size], self.lower_bound, self.upper_bound)
            
            inertia_weight *= self.inertia_damping  # Update local cached inertia weight