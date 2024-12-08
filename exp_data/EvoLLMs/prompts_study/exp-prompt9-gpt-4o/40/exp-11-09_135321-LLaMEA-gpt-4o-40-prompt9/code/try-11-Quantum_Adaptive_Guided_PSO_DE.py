import numpy as np

class Quantum_Adaptive_Guided_PSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Particle Swarm Optimization with Quantum-Inspired and Adaptive Guidance
        self.num_particles = 50
        self.inertia_weight = 0.6  # Dynamically adjusted
        self.cognitive_coeff = 1.7
        self.social_coeff = 1.4

        # Differential Evolution with Variable Control Parameters
        self.F_min, self.F_max = 0.5, 0.9
        self.CR_min, self.CR_max = 0.7, 1.0

        # Initialize particles in quantum space
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def adaptive_parameters(self, progress):
        """Adaptively adjust parameters based on optimization progress."""
        F = self.F_min + (self.F_max - self.F_min) * progress
        CR = self.CR_max - (self.CR_max - self.CR_min) * progress
        return F, CR

    def quantum_particle_update(self, r1, r2, i):
        """Update velocity using quantum-inspired adaptation."""
        alpha = np.random.uniform(0.5, 1.0)
        return alpha * (self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.positions[i]) + 
                        self.social_coeff * r2 * (self.global_best_position - self.positions[i]))

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

            # Update velocities and positions (Quantum-Inspired PSO)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            for i in range(self.num_particles):
                self.velocities[i] = self.inertia_weight * self.velocities[i] + self.quantum_particle_update(r1[i], r2[i], i)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Perform Differential Evolution with variable F and CR
            progress = evals / self.budget
            F, CR = self.adaptive_parameters(progress)
            for i in range(self.num_particles):
                indices = [idx for idx in range(self.num_particles) if idx != i]
                x1, x2, x3 = self.positions[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(x1 + F * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < CR, mutant_vector, self.positions[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                trial_score = func(trial_vector)

                # DE acceptance criterion
                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
                    
            evals += self.num_particles

        return self.global_best_position, self.global_best_score