import numpy as np

class Hybrid_DQPSO_EDA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # DQPSO parameters
        self.num_particles = 40
        self.inertia_weight = 0.5  # Increased for better exploration
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.6

        # Estimation of Distribution parameters
        self.alpha = 0.1  # Learning rate for distribution update

        # Initialize particles in quantum space
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        evals = 0
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

            # Update velocities and positions (DQPSO)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
            self.positions += self.velocities * np.random.uniform(0.1, 0.5, self.positions.shape)
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Estimation of Distribution (EDA)
            mean = np.mean(self.positions, axis=0)
            stddev = np.std(self.positions, axis=0)
            new_samples = np.random.normal(mean, stddev, self.positions.shape)
            new_samples = np.clip(new_samples, self.lower_bound, self.upper_bound)

            # Update positions based on EDA samples
            for i in range(self.num_particles):
                if evals < self.budget:
                    trial_score = func(new_samples[i])
                    evals += 1
                    if trial_score < scores[i]:
                        self.positions[i] = new_samples[i]
                        scores[i] = trial_score

        return self.global_best_position, self.global_best_score