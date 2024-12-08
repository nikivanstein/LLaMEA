import numpy as np

class Enhanced_Hybrid_PSO_DE_LF:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Enhanced Multi-Swarm PSO parameters
        self.num_particles = 50  # Slightly increased for more exploration
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5  # Adjusted for balanced exploration-exploitation
        self.social_coeff = 1.7

        # Differential Evolution parameters
        self.F = 0.8  # Adjusted scaling factor
        self.CR = 0.8  # Adjusted crossover rate

        # Particle initialization
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim)) * 0.1
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def chaotic_inertia(self, iter_count):
        return 0.5 + 0.5 * np.cos(3 * np.pi * iter_count / self.budget)

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, L)
        v = np.random.normal(0, 1, L)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def select_parents(self, scores):
        idxs = np.random.choice(range(len(scores)), 3, replace=False)
        return idxs

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

            # Update velocities and positions (PSO)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = (self.chaotic_inertia(iter_count) * self.velocities +
                               cognitive_component + social_component)
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Differential Evolution with Lévy flights
            for i in range(self.num_particles):
                idx1, idx2, idx3 = self.select_parents(scores)
                x1, x2, x3 = self.positions[idx1], self.positions[idx2], self.positions[idx3]
                mutant_vector = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, self.positions[i])
                
                # Incorporate Lévy flights for exploration
                levy_steps = self.levy_flight(self.dim)
                trial_vector += 0.01 * levy_steps * (trial_vector - self.positions[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                
                trial_score = func(trial_vector)

                # DE acceptance criterion
                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
            
            evals += self.num_particles
            iter_count += 1

        return self.global_best_position, self.global_best_score