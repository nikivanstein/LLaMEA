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
        self.global_best_position = np.copy(self.particles[0])
        self.global_best_score = np.inf
        self.evaluations = 0
        self.random_matrix = np.random.rand(self.population_size, self.dim, 2)

    def __call__(self, func):
        score_evaluation_batch = np.zeros(self.population_size)
        while self.evaluations < self.budget:
            # Batch score evaluation for efficiency
            for i in range(self.population_size):
                score_evaluation_batch[i] = func(self.particles[i])
            scores = score_evaluation_batch
            self.evaluations += self.population_size

            better_mask = scores < self.personal_best_scores
            np.copyto(self.personal_best_scores, scores, where=better_mask)
            np.copyto(self.personal_best_positions, self.particles, where=better_mask[:, None])
            min_score_index = np.argmin(scores)
            if scores[min_score_index] < self.global_best_score:
                self.global_best_score = scores[min_score_index]
                self.global_best_position[:] = self.particles[min_score_index]

            # Unrolling cognitive and social update to avoid repeated computations
            cognitive_component = (self.cognitive_const * self.random_matrix[:, :, 0] * 
                                   (self.personal_best_positions - self.particles))
            social_component = (self.social_const * self.random_matrix[:, :, 1] * 
                                (self.global_best_position - self.particles))
            self.velocities = (self.inertia_weight * self.velocities + 
                               cognitive_component + 
                               social_component)

            np.clip(self.velocities, -1, 1, out=self.velocities)
            self.particles += self.velocities
            np.clip(self.particles, -5, 5, out=self.particles)

        return self.global_best_position, self.global_best_score