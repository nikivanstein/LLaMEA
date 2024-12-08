import numpy as np

class OptimizedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.inertia_weight = 0.729
        self.cognitive_const = 1.49445
        self.social_const = 1.49445
        self.particles = np.random.uniform(-5, 5, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        rng = np.random.default_rng()  # Use NumPy's random generator for efficiency
        update_cycle = 5  # Decide upon a less frequent full update cycle

        while self.evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, self.particles[:min(self.population_size, self.budget - self.evaluations)])
            self.evaluations += len(scores)

            # Update personal and global bests
            improved = scores < self.personal_best_scores[:len(scores)]
            self.personal_best_scores[:len(scores)][improved] = scores[improved]
            self.personal_best_positions[:len(scores)][improved] = self.particles[:len(scores)][improved]

            if np.min(scores) < self.global_best_score:
                self.global_best_score = np.min(scores)
                self.global_best_position = self.particles[np.argmin(scores)]

            # Update velocities and positions every update_cycle iterations
            if self.evaluations % update_cycle == 0 or self.evaluations >= self.budget:
                cognitive_component = self.cognitive_const * rng.random((self.population_size, self.dim)) * (self.personal_best_positions - self.particles)
                social_component = self.social_const * rng.random((self.population_size, self.dim)) * (self.global_best_position - self.particles)

                self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
                self.velocities = np.clip(self.velocities, -1, 1)

                self.particles += self.velocities
                self.particles = np.clip(self.particles, -5, 5)

        return self.global_best_position, self.global_best_score