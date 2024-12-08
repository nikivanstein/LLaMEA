import numpy as np

class Adaptive_Hybrid_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Quantum Particle Swarm parameters
        self.num_particles = 50  # Increased to enhance exploration
        self.inertia_weight = 0.5  # Fine-tuned inertia weight
        self.cognitive_coeff = 1.6
        self.social_coeff = 1.8

        # Differential Evolution parameters
        self.F = 0.8  # Adjusted for more aggressive search
        self.CR = 0.85  # Fine-tuned crossover probability

        # Initialize particles in quantum space
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-0.3, 0.3, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def levy_flight(self, L):
        return np.random.standard_cauchy(size=L)

    def chemotaxis(self, position, func):
        step_size = 0.01
        direction = np.random.uniform(-1, 1, self.dim)
        new_position = position + step_size * direction
        new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
        new_score = func(new_position)
        return new_position if new_score < func(position) else position

    def __call__(self, func):
        evals = 0
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
            self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
            self.positions += self.velocities * np.random.uniform(0.2, 0.6, self.positions.shape)
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            for i in range(self.num_particles):
                indices = [idx for idx in range(self.num_particles) if idx != i]
                x1, x2, x3 = self.positions[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, self.positions[i])
                
                levy_steps = self.levy_flight(self.dim)
                trial_vector += 0.02 * levy_steps * (trial_vector - self.positions[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                
                trial_score = func(trial_vector)

                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
                
                # Bacterial Foraging-inspired chemotaxis
                self.positions[i] = self.chemotaxis(self.positions[i], func)
                scores[i] = func(self.positions[i])

            evals += self.num_particles

        return self.global_best_position, self.global_best_score