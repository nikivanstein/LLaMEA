import numpy as np

class Quantum_Evolutionary_Swarm_Optimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # QPSO parameters
        self.num_particles = 30  # Reduced number of particles for faster convergence
        self.inertia_weight = 0.5  # Adjusted dynamic inertia weight
        self.cognitive_coeff = 1.2
        self.social_coeff = 1.6

        # Genetic Algorithm parameters
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8

        # Initialize particles
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-0.3, 0.3, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def strategic_mutation(self, individual):
        return individual + self.mutation_rate * np.random.normal(size=self.dim)

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

            # Update velocities and positions (QPSO)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Apply Genetic Algorithm-inspired mutation and crossover
            for i in range(self.num_particles):
                if np.random.rand() < self.crossover_rate:
                    partner_idx = np.random.randint(self.num_particles)
                    partner = self.positions[partner_idx]
                    crossover_point = np.random.randint(1, self.dim)
                    trial_vector = np.concatenate((self.positions[i][:crossover_point], partner[crossover_point:]))
                    trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                    
                    if np.random.rand() < self.mutation_rate:
                        trial_vector = self.strategic_mutation(trial_vector)
                        trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                    
                    trial_score = func(trial_vector)

                    # Acceptance criterion
                    if trial_score < scores[i]:
                        self.positions[i] = trial_vector
                        scores[i] = trial_score

            evals += self.num_particles

        return self.global_best_position, self.global_best_score