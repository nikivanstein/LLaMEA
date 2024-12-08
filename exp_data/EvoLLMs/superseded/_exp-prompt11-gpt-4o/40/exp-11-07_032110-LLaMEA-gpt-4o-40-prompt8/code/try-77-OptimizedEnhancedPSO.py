import numpy as np

class OptimizedEnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(30, self.budget // 10)  # Adapt particle count to budget
        self.inertia_weight = 0.729
        self.cognitive_const = 1.49445
        self.social_const = 1.49445
        self.particles = np.random.uniform(-5, 5, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))  # Start with zero velocities
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = np.copy(self.particles[0])
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, self.particles)
            self.evaluations += self.population_size

            better_mask = scores < self.personal_best_scores
            np.copyto(self.personal_best_scores, scores, where=better_mask)
            np.copyto(self.personal_best_positions, self.particles, where=better_mask)

            min_score_index = np.argmin(scores)
            if scores[min_score_index] < self.global_best_score:
                self.global_best_score = scores[min_score_index]
                self.global_best_position = np.copy(self.particles[min_score_index])

            rand1, rand2 = np.random.rand(2, self.population_size, self.dim)

            self.velocities = (
                self.inertia_weight * self.velocities +
                self.cognitive_const * rand1 * (self.personal_best_positions - self.particles) +
                self.social_const * rand2 * (self.global_best_position - self.particles)
            )
            np.clip(self.velocities, -1, 1, out=self.velocities)
            self.particles += self.velocities
            np.clip(self.particles, -5, 5, out=self.particles)

        return self.global_best_position, self.global_best_score