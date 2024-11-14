import numpy as np

class Enhanced_Multi_heuristic_Optimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Particle Swarm Optimization parameters
        self.num_particles = 40  # Increased particle count for better exploration
        self.inertia_weight = 0.7  # Dynamic inertia for balance
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5

        # Adaptive Mutation Strategy parameters
        self.F_base = 0.9  # Slightly higher base scaling factor
        self.CR_base = 0.7  # Lower base crossover probability

        # Particle initializations
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def logistic_map(self, x):
        return 4.0 * x * (1 - x)  # Logistic map for chaotic sequence

    def adaptive_mutation(self, current_iter, total_iters):
        return self.F_base + 0.3 * (1 - current_iter / total_iters)

    def quantum_perturbation(self, L):
        return np.random.normal(0, 1, L)

    def __call__(self, func):
        evals = 0
        chaos_factor = np.random.rand()
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

            # Update velocities and positions (Guided PSO)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = (self.inertia_weight * self.velocities + cognitive_component + social_component) * chaos_factor
            self.positions += self.velocities * np.random.uniform(0.2, 0.6, self.positions.shape)
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Adaptive Mutation with Quantum-inspired Dynamics
            F = self.adaptive_mutation(iter_count, self.budget)
            for i in range(self.num_particles):
                indices = [idx for idx in range(self.num_particles) if idx != i]
                x1, x2, x3 = self.positions[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(x1 + F * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR_base, mutant_vector, self.positions[i])
                
                # Incorporate Quantum perturbation for diverse exploration
                quantum_steps = self.quantum_perturbation(self.dim)
                trial_vector += 0.02 * quantum_steps * (trial_vector - self.positions[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                
                trial_score = func(trial_vector)

                # DE acceptance criterion
                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
            
            chaos_factor = self.logistic_map(chaos_factor)  # Update chaos factor for next iteration
            evals += self.num_particles
            iter_count += 1

        return self.global_best_position, self.global_best_score