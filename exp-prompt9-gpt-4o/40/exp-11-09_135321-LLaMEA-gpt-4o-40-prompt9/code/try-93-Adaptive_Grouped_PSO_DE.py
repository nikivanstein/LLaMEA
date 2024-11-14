import numpy as np

class Adaptive_Grouped_PSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Parameters
        self.num_particles = 50  # Increased particle count
        self.inertia_weight = 0.6  # Adaptive inertia weight
        self.cognitive_coeff = 2.0
        self.social_coeff = 1.3

        # Differential Evolution parameters
        self.F_base = 0.8  # Enhanced scaling factor
        self.CR_base = 0.85  # Balanced crossover probability

        # Particle initializations
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def adaptive_inertia(self, iter_count):
        return 0.6 + 0.4 * np.cos(2 * np.pi * iter_count / self.budget)

    def adaptive_mutation_factor(self, current_iter, total_iters):
        return self.F_base + 0.2 * (current_iter / total_iters)

    def gaussian_mutation(self, L):
        return np.random.normal(0, 0.1, L)

    def dynamic_grouping(self, scores):
        return np.argpartition(scores, 3)[:3]

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

            # Update velocities and positions
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = (self.adaptive_inertia(iter_count) * self.velocities +
                               cognitive_component + social_component)
            self.positions += self.velocities * np.random.uniform(0.1, 0.2, self.positions.shape)
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Differential Evolution with Gaussian mutation
            F = self.adaptive_mutation_factor(iter_count, self.budget)
            for i in range(self.num_particles):
                idx1, idx2, idx3 = self.dynamic_grouping(scores)
                x1, x2, x3 = self.positions[idx1], self.positions[idx2], self.positions[idx3]
                mutant_vector = np.clip(x1 + F * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR_base, mutant_vector, self.positions[i])
                
                # Gaussian mutation for exploration
                gauss_steps = self.gaussian_mutation(self.dim)
                trial_vector += 0.05 * gauss_steps * (trial_vector - self.positions[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                
                trial_score = func(trial_vector)

                # DE acceptance criterion
                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
            
            evals += self.num_particles
            iter_count += 1

        return self.global_best_position, self.global_best_score