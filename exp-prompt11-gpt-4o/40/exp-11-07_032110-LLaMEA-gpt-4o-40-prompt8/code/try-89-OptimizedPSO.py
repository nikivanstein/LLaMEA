import numpy as np

class OptimizedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 30  # Reduced initial population size
        self.inertia_weight = 0.7
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.particles = np.random.uniform(-5, 5, (self.initial_population_size, self.dim))
        self.velocities = np.zeros((self.initial_population_size, self.dim))  # Initialize velocities to zero
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.initial_population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, self.particles)
            self.evaluations += len(self.particles)

            better_mask = scores < self.personal_best_scores
            self.personal_best_scores[better_mask] = scores[better_mask]
            self.personal_best_positions[better_mask] = self.particles[better_mask]
            
            min_score_index = np.argmin(scores)
            if scores[min_score_index] < self.global_best_score:
                self.global_best_score = scores[min_score_index]
                self.global_best_position = self.particles[min_score_index].copy()

            if self.global_best_score < 1e-8:  # Early stopping condition
                break

            random_matrix1, random_matrix2 = np.random.rand(self.initial_population_size, self.dim), np.random.rand(self.initial_population_size, self.dim)

            cognitive_component = self.cognitive_const * random_matrix1 * (self.personal_best_positions - self.particles)
            social_component = self.social_const * random_matrix2 * (self.global_best_position - self.particles)

            self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
            self.particles += self.velocities
            np.clip(self.particles, -5, 5, out=self.particles)

            # Adaptive population resizing
            if self.evaluations < self.budget * 0.5:
                self.adapt_population_size()

        return self.global_best_position, self.global_best_score

    def adapt_population_size(self):
        new_size = min(self.initial_population_size * 2, int(self.budget / 10))
        if new_size > len(self.particles):
            additional_particles = np.random.uniform(-5, 5, (new_size - len(self.particles), self.dim))
            self.particles = np.vstack((self.particles, additional_particles))
            self.velocities = np.vstack((self.velocities, np.zeros((new_size - len(self.velocities), self.dim))))
            self.personal_best_positions = np.vstack((self.personal_best_positions, additional_particles))
            self.personal_best_scores = np.append(self.personal_best_scores, np.full(new_size - len(self.personal_best_scores), np.inf))