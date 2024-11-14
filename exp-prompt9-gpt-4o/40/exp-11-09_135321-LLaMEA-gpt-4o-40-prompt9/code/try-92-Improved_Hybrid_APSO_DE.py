import numpy as np

class Improved_Hybrid_APSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Enhanced Multi-Swarm APSO parameters
        self.num_particles = 50  # Adjusted particle count for balance
        self.inertia_weight = 0.6  # Adjusted inertia weight for faster convergence
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.7

        # Adaptive Differential Evolution parameters
        self.F_base = 0.8  # Adjusted scaling factor for better exploration
        self.CR_base = 0.85  # Adjusted crossover probability for diversity

        # Particle initializations
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def dynamic_inertia(self, iter_count):
        return 0.4 + 0.5 * np.sin(2 * np.pi * iter_count / self.budget)

    def adaptive_mutation_strategy(self, current_iter, total_iters):
        return self.F_base + 0.4 * np.cos(np.pi * current_iter / total_iters)

    def spiral_search(self, L):
        alpha = 1.1
        theta = np.random.uniform(0, 2 * np.pi, L)
        return alpha * np.cos(theta), alpha * np.sin(theta)

    def rank_based_selection(self, scores):
        ranks = np.argsort(np.argsort(scores))
        probabilities = (len(scores) - ranks) / sum(len(scores) - ranks)
        return np.random.choice(range(len(scores)), p=probabilities)

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

            # Update velocities and positions (Enhanced Multi-Swarm APSO)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = (self.dynamic_inertia(iter_count) * self.velocities +
                               cognitive_component + social_component)
            self.positions += self.velocities * np.random.uniform(0.1, 0.3, self.positions.shape)
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Adaptive Differential Evolution with Spiral-guided search
            F = self.adaptive_mutation_strategy(iter_count, self.budget)
            for i in range(self.num_particles):
                idx1, idx2, idx3 = [self.rank_based_selection(scores) for _ in range(3)]
                x1, x2, x3 = self.positions[idx1], self.positions[idx2], self.positions[idx3]
                mutant_vector = np.clip(x1 + F * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR_base, mutant_vector, self.positions[i])
                
                # Incorporate Spiral-guided search for better exploration
                spiral_x, spiral_y = self.spiral_search(self.dim)
                trial_vector += 0.01 * np.array([spiral_x, spiral_y]).T * (trial_vector - self.positions[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                
                trial_score = func(trial_vector)

                # DE acceptance criterion
                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
            
            evals += self.num_particles
            iter_count += 1

        return self.global_best_position, self.global_best_score