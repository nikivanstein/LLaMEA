import numpy as np

class Enhanced_Hybrid_APSO_DE_v2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Parameters
        self.num_particles = 50  # Adjusted for better diversity
        self.elite_archive_size = 5
        self.inertia_weight = 0.6  # Dynamic inertia weight
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.7

        # DE parameters
        self.F_base = 0.8  # Enhanced exploration-exploitation balance
        self.CR_base = 0.85

        # Initialization
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best and elite archive
        self.global_best_position = None
        self.global_best_score = np.inf
        self.elite_archive = []

    def chaotic_inertia(self, iter_count):
        return 0.4 + 0.5 * np.sin(4 * np.pi * iter_count / self.budget)

    def adaptive_mutation_strategy(self, current_iter, total_iters):
        return self.F_base + 0.2 * (current_iter / total_iters)

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, L)
        v = np.random.normal(0, 1, L)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def update_elite_archive(self, scores):
        sorted_indices = np.argsort(scores)
        for idx in sorted_indices[:self.elite_archive_size]:
            if len(self.elite_archive) < self.elite_archive_size:
                self.elite_archive.append(self.positions[idx])
            else:
                if scores[idx] < func(self.elite_archive[-1]):
                    self.elite_archive[-1] = self.positions[idx]
            self.elite_archive.sort(key=func)

    def __call__(self, func):
        evals = 0
        iter_count = 0

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

            # Update velocities and positions with elite influence
            self.update_elite_archive(scores)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            elite_influence = np.mean(self.elite_archive, axis=0) if self.elite_archive else self.global_best_position
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (elite_influence - self.positions)
            self.velocities = (self.chaotic_inertia(iter_count) * self.velocities +
                               cognitive_component + social_component)
            self.positions += self.velocities * np.random.uniform(0.1, 0.3, self.positions.shape)
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Adaptive Differential Evolution with Lévy flights
            F = self.adaptive_mutation_strategy(iter_count, self.budget)
            for i in range(self.num_particles):
                idx1, idx2 = np.random.choice(range(self.num_particles), 2, replace=False)
                x1, x2, x3 = self.positions[idx1], self.positions[idx2], self.global_best_position
                mutant_vector = np.clip(x1 + F * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR_base, mutant_vector, self.positions[i])
                
                # Incorporate Levy flights for better exploration
                levy_steps = self.levy_flight(self.dim)
                trial_vector += 0.02 * levy_steps * (trial_vector - self.positions[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                
                trial_score = func(trial_vector)

                # DE acceptance criterion
                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
            
            evals += self.num_particles
            iter_count += 1

        return self.global_best_position, self.global_best_score