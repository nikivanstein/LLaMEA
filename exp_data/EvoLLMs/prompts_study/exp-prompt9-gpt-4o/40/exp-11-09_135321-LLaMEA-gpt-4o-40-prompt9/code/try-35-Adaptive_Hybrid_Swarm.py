import numpy as np

class Adaptive_Hybrid_Swarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Swarm parameters
        self.num_particles = 60  # Increased population for diversity
        self.inertia_weight = 0.7  # Dynamic inertia weight for balance
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.8

        # Differential Evolution parameters
        self.F = 0.9  # Higher scaling factor for stronger perturbation
        self.CR = 0.9  # Increased crossover probability

        # Initialize particles, velocities and memory
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)
        self.memory = np.copy(self.positions) # Memory for quantum-inspired update

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def chaotic_map(self, x):
        return 4 * x * (1 - x)  # Logistic map for chaotic sequence

    def levy_flight(self, L):
        return np.random.standard_cauchy(size=L)

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

            # Update velocities and positions (Hybrid Swarm)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = (self.inertia_weight * self.velocities + cognitive_component + social_component) * chaos_factor
            self.positions += self.velocities * np.random.uniform(0.2, 0.4, self.positions.shape)
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Quantum-inspired Memory Update
            quantum_update = self.memory + np.random.uniform(-1, 1, (self.num_particles, self.dim)) * (self.positions - self.memory)
            self.positions = np.where(np.random.rand(self.num_particles, self.dim) < 0.5, quantum_update, self.positions)
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Perform Differential Evolution with Lévy flights
            for i in range(self.num_particles):
                indices = [idx for idx in range(self.num_particles) if idx != i]
                x1, x2, x3 = self.positions[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)
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