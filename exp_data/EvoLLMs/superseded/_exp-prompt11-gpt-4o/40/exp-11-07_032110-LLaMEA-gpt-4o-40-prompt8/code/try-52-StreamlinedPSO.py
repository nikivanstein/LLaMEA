import numpy as np

class StreamlinedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Reduced population size
        self.inertia_weight = 0.7  # Slightly adjusted inertia weight
        self.cognitive_const = 1.5
        self.social_const = 1.5
        # Initialize particles using a more efficient method
        self.particles = np.random.uniform(-5, 5, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = np.copy(self.particles[0])
        self.global_best_score = np.inf
        self.evaluations = 0

    def update_particles(self, func):
        scores = np.apply_along_axis(func, 1, self.particles)
        self.evaluations += self.population_size

        better_mask = scores < self.personal_best_scores
        np.copyto(self.personal_best_scores, scores, where=better_mask)
        np.copyto(self.personal_best_positions, self.particles, where=better_mask[:, None])

        min_score_index = np.argmin(scores)
        if scores[min_score_index] < self.global_best_score:
            self.global_best_score = scores[min_score_index]
            self.global_best_position[:] = self.particles[min_score_index]

        return scores

    def __call__(self, func):
        random_matrix = np.random.rand(self.population_size, self.dim, 2)
        while self.evaluations < self.budget:
            scores = self.update_particles(func)

            cognitive_component = self.cognitive_const * random_matrix[:, :, 0] * (self.personal_best_positions - self.particles)
            social_component = self.social_const * random_matrix[:, :, 1] * (self.global_best_position - self.particles)

            self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component

            np.clip(self.velocities, -1, 1, out=self.velocities)
            self.particles += self.velocities
            np.clip(self.particles, -5, 5, out=self.particles)

            # Local search integration to enhance exploitation
            if self.evaluations < self.budget:
                local_best_pos = self.particles[np.argmin(scores)]
                self.particles = np.clip(self.particles + np.random.uniform(-0.1, 0.1, self.particles.shape), -5, 5)
                self.particles[:, :] = np.where(scores[:, None] < self.personal_best_scores[:, None], local_best_pos, self.particles)

        return self.global_best_position, self.global_best_score