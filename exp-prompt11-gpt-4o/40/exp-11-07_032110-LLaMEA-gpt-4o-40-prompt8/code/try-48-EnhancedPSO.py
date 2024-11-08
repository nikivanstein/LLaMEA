import numpy as np

class EnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.inertia_weight = 0.729
        self.cognitive_const = 1.49445
        self.social_const = 1.49445
        self.particles = np.random.uniform(-5, 5, (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = np.copy(self.particles[0])
        self.global_best_score = np.inf
        self.evaluations = 0
        self.max_velocity = 0.1 * (5 - (-5))

    def __call__(self, func):
        while self.evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, self.particles)
            self.evaluations += self.population_size

            better_mask = scores < self.personal_best_scores
            self.personal_best_scores = np.where(better_mask, scores, self.personal_best_scores)
            self.personal_best_positions = np.where(better_mask[:, None], self.particles, self.personal_best_positions)

            min_score_index = np.argmin(scores)
            if scores[min_score_index] < self.global_best_score:
                self.global_best_score = scores[min_score_index]
                self.global_best_position = np.copy(self.particles[min_score_index])

            random_matrix = np.random.rand(self.population_size, self.dim)
            cognitive_component = (self.cognitive_const * random_matrix *
                                   (self.personal_best_positions - self.particles))
            social_component = (self.social_const * random_matrix *
                                (self.global_best_position - self.particles))

            self.velocities = (self.inertia_weight * self.velocities +
                               cognitive_component + 
                               social_component)

            np.clip(self.velocities, -self.max_velocity, self.max_velocity, out=self.velocities)
            self.particles += self.velocities
            np.clip(self.particles, -5, 5, out=self.particles)

        return self.global_best_position, self.global_best_score