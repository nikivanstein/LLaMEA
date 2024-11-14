import numpy as np

class DynamicMutationPSO:
    def __init__(self, budget, dim, initial_particles=30, inertia_weight=0.7, cognitive_param=1.5, social_param=1.5, velocity_clamp=0.5):
        self.budget = budget
        self.dim = dim
        self.inertia_weight = inertia_weight
        self.cognitive_param = cognitive_param
        self.social_param = social_param
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocity_clamp = velocity_clamp
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (initial_particles, dim))
        self.velocities = np.random.uniform(-1, 1, (initial_particles, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(initial_particles, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_score = np.inf

    def __call__(self, func):
        evaluations = 0
        num_particles = self.positions.shape[0]
        
        while evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, self.positions)
            evaluations += num_particles

            for i in range(num_particles):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]

                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]

            progress_factor = evaluations / self.budget
            dynamic_inertia_weight = self.inertia_weight * (1 - progress_factor) * 0.99
            diversity = np.mean(np.std(self.positions, axis=0))
            adaptive_cognitive_param = self.cognitive_param + 0.5 * (1.0 - diversity / (self.upper_bound - self.lower_bound))
            adaptive_social_param = self.social_param + 0.5 * (diversity / (self.upper_bound - self.lower_bound))
            
            if evaluations % (self.budget // 10) == 0 and num_particles < 50:
                # Add more particles overtime
                additional_particles = np.random.uniform(self.lower_bound, self.upper_bound, (5, self.dim))
                self.positions = np.vstack([self.positions, additional_particles])
                self.velocities = np.vstack([self.velocities, np.random.uniform(-1, 1, (5, self.dim))])
                self.personal_best_positions = np.vstack([self.personal_best_positions, additional_particles])
                self.personal_best_scores = np.concatenate([self.personal_best_scores, np.full(5, np.inf)])
                num_particles = self.positions.shape[0]

            neighborhood_best_positions = self.positions[np.argpartition(scores, 3)[:3]]
            neighborhood_best = np.mean(neighborhood_best_positions, axis=0)

            self.velocities = (
                dynamic_inertia_weight * self.velocities +
                adaptive_cognitive_param * np.random.rand(num_particles, self.dim) * (self.personal_best_positions - self.positions) +
                adaptive_social_param * np.random.rand(num_particles, self.dim) * (self.global_best_position - self.positions) +
                0.15 * np.random.rand(num_particles, self.dim) * (neighborhood_best - self.positions)
            )
            self.velocities = np.clip(self.velocities, -self.velocity_clamp, self.velocity_clamp)
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)
            
            # Introduce mutation for further exploration
            if evaluations % (self.budget // 5) == 0:
                mutation_indices = np.random.choice(num_particles, int(0.1 * num_particles), replace=False)
                self.positions[mutation_indices] += np.random.normal(0, 0.1, (len(mutation_indices), self.dim))
                self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score