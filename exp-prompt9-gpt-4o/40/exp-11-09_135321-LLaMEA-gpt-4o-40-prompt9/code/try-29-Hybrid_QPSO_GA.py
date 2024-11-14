import numpy as np

class Hybrid_QPSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
        # Parameters
        self.num_particles = 50
        self.inertia_weight = 0.7  # Adjusted inertia weight
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.8
        
        # Genetic Algorithm parameters
        self.mutation_rate = 0.02  # Adaptive mutation rate
        self.crossover_rate = 0.9
        
        # Initialize particles
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)
        
        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def chaotic_map(self, x):
        return 4 * x * (1 - x)

    def levy_flight(self, L):
        return np.random.standard_cauchy(size=L)

    def adaptive_mutation(self, particle, global_best):
        if np.random.rand() < self.mutation_rate:
            mutation_vector = np.random.standard_normal(size=self.dim)
            return np.clip(particle + mutation_vector * (global_best - particle), self.lower_bound, self.upper_bound)
        return particle

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

            # Update velocities and positions (Enhanced QPSO)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = (self.inertia_weight * self.velocities + cognitive_component + social_component) * chaos_factor
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)
            
            # Genetic Algorithm crossover and mutation
            for i in range(0, self.num_particles, 2):
                if i + 1 < self.num_particles and np.random.rand() < self.crossover_rate:
                    crossover_point = np.random.randint(1, self.dim)
                    self.positions[i, crossover_point:], self.positions[i + 1, crossover_point:] = (
                        self.positions[i + 1, crossover_point:], self.positions[i, crossover_point:]
                    )

            for i in range(self.num_particles):
                mutated_particle = self.adaptive_mutation(self.positions[i], self.global_best_position)
                mutated_score = func(mutated_particle)
                if mutated_score < scores[i]:
                    self.positions[i] = mutated_particle
                    scores[i] = mutated_score
            
            chaos_factor = self.chaotic_map(chaos_factor)
            evals += self.num_particles

        return self.global_best_position, self.global_best_score