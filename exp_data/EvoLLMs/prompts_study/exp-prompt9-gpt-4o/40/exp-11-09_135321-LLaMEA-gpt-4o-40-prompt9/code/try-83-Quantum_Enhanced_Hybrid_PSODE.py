import numpy as np

class Quantum_Enhanced_Hybrid_PSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Quantum-inspired PSO parameters
        self.num_particles = 50  # Increased particle count for enhanced exploration
        self.inertia_weight = 0.6  # Dynamically adjusting inertia weight
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.8

        # Adaptive Differential Evolution parameters
        self.F_base = 0.8  # Altered scaling factor for exploration/exploitation balance
        self.CR_base = 0.85  # Variation in crossover probability

        # Particle initializations
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def chaotic_inertia(self, iter_count):
        return 0.4 + 0.5 * np.sin(2 * np.pi * iter_count / self.budget)  # Leveraging chaos theory

    def adaptive_mutation_strategy(self, current_iter, total_iters):
        return self.F_base + 0.2 * (1 - current_iter / total_iters)

    def quantum_levy_flight(self, L):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, L)
        v = np.random.normal(0, 1, L)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def tournament_selection(self, scores, size=3):
        indices = np.random.choice(range(len(scores)), size, replace=False)
        return indices[np.argmin(scores[indices])]

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

            # Update velocities and positions (Quantum-inspired PSO)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = (self.chaotic_inertia(iter_count) * self.velocities +
                               cognitive_component + social_component)
            self.positions += self.velocities * np.random.uniform(0.15, 0.35, self.positions.shape)
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Adaptive Differential Evolution with Quantum Lévy flights
            F = self.adaptive_mutation_strategy(iter_count, self.budget)
            for i in range(self.num_particles):
                idx1, idx2, idx3 = [self.tournament_selection(scores) for _ in range(3)]
                x1, x2, x3 = self.positions[idx1], self.positions[idx2], self.positions[idx3]
                mutant_vector = np.clip(x1 + F * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR_base, mutant_vector, self.positions[i])
                
                # Incorporate Quantum Levy flights for better exploration
                quantum_steps = self.quantum_levy_flight(self.dim)
                trial_vector += 0.02 * quantum_steps * (trial_vector - self.positions[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                
                trial_score = func(trial_vector)

                # DE acceptance criterion
                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
            
            evals += self.num_particles
            iter_count += 1

        return self.global_best_position, self.global_best_score