import numpy as np

class AQPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Adaptive Quantum Particle Swarm Optimization parameters
        self.num_particles = 50  # Increased particle count for more diversity
        self.inertia_weight = 0.5
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.quantum_delta = 0.005  # Quantum step size for exploration

        # Differential Evolution parameters
        self.F_base = 0.6  # Scaling factor for DE mutation
        self.CR_base = 0.8  # Crossover probability

        # Particle initializations
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def quantum_inspired_motion(self, position, best_position):
        return position + self.quantum_delta * np.sign(np.random.rand(self.dim) - 0.5) * (best_position - position)

    def adaptive_mutation_strategy(self, current_iter, total_iters):
        return self.F_base + 0.2 * np.sin(2 * np.pi * current_iter / total_iters)

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

            # Update velocities and positions (AQPSO)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = (self.inertia_weight * self.velocities +
                               cognitive_component + social_component)
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Apply quantum-inspired motion for enhanced exploration
            if iter_count % 10 == 0:  # Apply periodically to prevent premature convergence
                for i in range(self.num_particles):
                    self.positions[i] = self.quantum_inspired_motion(self.positions[i], self.global_best_position)

            # Differential Evolution
            F = self.adaptive_mutation_strategy(iter_count, self.budget)
            for i in range(self.num_particles):
                idx1, idx2, idx3 = np.random.choice(range(self.num_particles), 3, replace=False)
                x1, x2, x3 = self.positions[idx1], self.positions[idx2], self.positions[idx3]
                mutant_vector = np.clip(x1 + F * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR_base, mutant_vector, self.positions[i])
                trial_score = func(trial_vector)

                # DE acceptance criterion
                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
            
            evals += self.num_particles
            iter_count += 1

        return self.global_best_position, self.global_best_score