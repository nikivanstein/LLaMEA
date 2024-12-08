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
        while self.evaluations < self.budget:
            # Evaluate the fitness of each particle
            scores = np.apply_along_axis(func, 1, self.particles)
            self.evaluations += len(scores)

            # Update personal and global bests
            better_scores = scores < self.personal_best_scores
            self.personal_best_scores[better_scores] = scores[better_scores]
            self.personal_best_positions[better_scores] = self.particles[better_scores]

            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < self.global_best_score:
                self.global_best_score = scores[min_score_idx]
                self.global_best_position = self.particles[min_score_idx]

            # Update velocities and positions
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            cognitive_component = self.cognitive_const * r1 * (self.personal_best_positions - self.particles)
            social_component = self.social_const * r2 * (self.global_best_position - self.particles)
            self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component

            # Ensure velocities and update particle positions
            self.velocities = np.clip(self.velocities, -1, 1)
            self.particles += self.velocities
            self.particles = np.clip(self.particles, -5, 5)

        return self.global_best_position, self.global_best_score