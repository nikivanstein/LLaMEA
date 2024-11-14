import numpy as np

class Hybrid_GIQS_ES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Enhanced parameters
        self.num_particles = 60  # Adjusted particle count
        self.inertia_weight = 0.6  # Tuned inertia weight
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.7

        # Evolution Strategy parameters
        self.step_size = 0.5  # Adaptive step size
        self.decay_rate = 0.99  # Decay for adaptive step size

        # Initialize particles and velocities
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def chaotic_map(self, x):
        return 4 * x * (1 - x)  # Logistic map for chaotic sequence

    def __call__(self, func):
        evals = 0
        chaos_factor = np.random.rand()

        while evals < self.budget:
            # Evaluate each particle
            scores = np.apply_along_axis(func, 1, self.positions)
            evals += self.num_particles

            # Update personal and global bests
            for i in range(self.num_particles):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]

                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]

            # Update velocities and positions (Hybrid GIQS)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = (self.inertia_weight * self.velocities + cognitive_component + social_component) * chaos_factor
            self.positions += self.velocities * self.step_size
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Perform Evolutionary Strategy with adaptive step size
            for i in range(self.num_particles):
                candidate_position = self.positions[i] + self.step_size * np.random.normal(0, 1, self.dim)
                candidate_position = np.clip(candidate_position, self.lower_bound, self.upper_bound)
                candidate_score = func(candidate_position)

                # Acceptance criterion
                if candidate_score < scores[i]:
                    self.positions[i] = candidate_position
                    scores[i] = candidate_score

            chaos_factor = self.chaotic_map(chaos_factor)  # Update chaos factor for next iteration
            self.step_size *= self.decay_rate  # Decay step size
            evals += self.num_particles

        return self.global_best_position, self.global_best_score