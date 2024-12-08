import numpy as np

class Enhanced_HCPSO_ALM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Parameters
        self.num_particles = 50  # Increased particle count for diversity
        self.inertia_weight = 0.6  # Adjusted inertia for adaptive learning
        self.cognitive_coeff = 1.7
        self.social_coeff = 1.4

        # Mutation parameters
        self.mutation_rate = 0.2  # Introduced mutation for enhanced exploration

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

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, L)
        v = np.random.normal(0, 1, L)
        step = u / np.abs(v) ** (1 / beta)
        return step

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

            # Update velocities and positions with mutation
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = (self.inertia_weight * self.velocities + cognitive_component + social_component) * chaos_factor

            # Mutation
            mutations = np.random.rand(self.num_particles, self.dim) < self.mutation_rate
            mutation_steps = np.random.uniform(-1, 1, (self.num_particles, self.dim))
            self.positions += self.velocities + mutations * mutation_steps
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Differential Evolution with Lévy flights
            for i in range(self.num_particles):
                indices = [idx for idx in range(self.num_particles) if idx != i]
                x1, x2, x3 = self.positions[np.random.choice(indices, 3, replace=False)]
                mutant_vector = x1 + self.inertia_weight * (x2 - x3)
                trial_vector = np.where(np.random.rand(self.dim) < 0.9, mutant_vector, self.positions[i])
                
                levy_steps = self.levy_flight(self.dim)
                trial_vector += 0.01 * levy_steps * (trial_vector - self.positions[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                
                trial_score = func(trial_vector)

                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
            
            chaos_factor = self.chaotic_map(chaos_factor)  # Update chaos factor
            evals += self.num_particles

        return self.global_best_position, self.global_best_score