import numpy as np

class Hybrid_QPSO_EM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Hybrid QPSO parameters
        self.num_particles = 60  # Increased particle count for better exploration
        self.inertia_weight = 0.7  # Slightly increased inertia weight for more exploration
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5

        # Evolutionary Mutation parameters
        self.F = 0.9  # Improved scaling factor
        self.CR = 0.9  # Increased crossover probability

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

    def levy_flight(self, L, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1/beta)
        u = np.random.normal(0, sigma, size=L)
        v = np.random.normal(0, 1, size=L)
        return u / np.abs(v)**(1/beta)

    def __call__(self, func):
        evals = 0
        chaos_factor = np.random.rand()
        
        while evals < self.budget:
            scores = np.apply_along_axis(func, 1, self.positions)
            evals += self.num_particles

            for i in range(self.num_particles):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]

                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]

            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = (self.inertia_weight * self.velocities + cognitive_component + social_component) * chaos_factor
            self.positions += self.velocities * np.random.uniform(0.1, 0.6, self.positions.shape)
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            for i in range(self.num_particles):
                indices = [idx for idx in range(self.num_particles) if idx != i]
                x1, x2, x3 = self.positions[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, self.positions[i])
                
                levy_steps = self.levy_flight(self.dim)
                trial_vector += 0.01 * levy_steps * (trial_vector - self.positions[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)

                trial_score = func(trial_vector)

                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
            
            chaos_factor = self.chaotic_map(chaos_factor)
            evals += self.num_particles

        return self.global_best_position, self.global_best_score