import numpy as np

class Quantum_Enhanced_APSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # APSO parameters
        self.num_particles = 50  # Increased number of particles
        self.inertia_weight = 0.5  # Adjusted dynamic inertia weight
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.8

        # Crossover mutation parameters
        self.mutation_rate = 0.1  # Lower mutation rate
        self.mutation_scale = 0.5  # Reduced scale for crossover

        # Initialize particles in quantum space
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        evals = 0
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

            # Update velocities and positions (APSO)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Perform crossover mutation
            for i in range(self.num_particles):
                if np.random.rand() < self.mutation_rate:
                    donor_idx = np.random.randint(0, self.num_particles)
                    donor_vector = self.positions[donor_idx]
                    self.positions[i] = (1 - self.mutation_scale) * self.positions[i] + self.mutation_scale * donor_vector
                    self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score