import numpy as np

class Hybrid_APSO_ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Hybrid APSO parameters
        self.num_particles = 30  # Reduced particle count for faster convergence
        self.inertia_weight = 0.8  # Adjusted inertia weight for more exploration
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.6

        # Adaptive Differential Evolution parameters
        self.F_min = 0.4
        self.F_max = 0.9  # Adaptive scaling for better search
        self.CR = 0.9

        # Initialize particles and velocities
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-0.3, 0.3, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def chaotic_map(self, x):
        return np.sin(np.pi * x)  # Sinusoidal map for better randomness

    def levy_flight(self, L):
        beta = 1.5
        u = np.random.normal(0, 1, L)
        v = np.random.normal(0, 1, L)
        return u / np.abs(v) ** (1 / beta)

    def adapt_F(self, current_eval):
        return self.F_min + (self.F_max - self.F_min) * (1 - current_eval / self.budget)

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

            # Update velocities and positions (Hybrid APSO)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = (self.inertia_weight * self.velocities + cognitive_component + social_component) * chaos_factor
            self.positions += self.velocities * np.random.uniform(0.15, 0.6, self.positions.shape)
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Perform Adaptive Differential Evolution with Lévy flights
            for i in range(self.num_particles):
                indices = [idx for idx in range(self.num_particles) if idx != i]
                x1, x2, x3 = self.positions[np.random.choice(indices, 3, replace=False)]
                F_adapted = self.adapt_F(evals)
                mutant_vector = np.clip(x1 + F_adapted * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, self.positions[i])

                # Incorporate Levy flights for better exploration
                levy_steps = self.levy_flight(self.dim)
                trial_vector += 0.01 * levy_steps * (trial_vector - self.positions[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)

                trial_score = func(trial_vector)

                # DE acceptance criterion
                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score

            chaos_factor = self.chaotic_map(chaos_factor)  # Update chaos factor for next iteration
            evals += self.num_particles

        return self.global_best_position, self.global_best_score