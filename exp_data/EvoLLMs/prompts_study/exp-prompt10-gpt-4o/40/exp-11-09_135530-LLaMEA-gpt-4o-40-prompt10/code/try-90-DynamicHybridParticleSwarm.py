import numpy as np

class DynamicHybridParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 60  # Increased population size
        self.c1 = 1.8  # Adjusted cognitive coefficient
        self.c2 = 1.8  # Adjusted social coefficient
        self.w_max = 0.9
        self.w_min = 0.4
        self.gamma = 0.15  # Modified memory-based velocity adjustment rate
        self.evaluations = 0

    def chaotic_initialization(self):
        return self.lb + (self.ub - self.lb) * (0.5 - np.random.rand(self.population_size, self.dim))

    def initialize_particles(self):
        particles = self.chaotic_initialization()
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_scores

    def differential_evolution(self, target_idx, best, particles):
        indices = np.random.choice(self.population_size, 3, replace=False)
        while target_idx in indices:
            indices = np.random.choice(self.population_size, 3, replace=False)
        r1, r2, r3 = particles[indices]
        mutated = r1 + 0.8 * (r2 - r3)  # Scale factor for differential evolution
        return np.clip((mutated + best) / 2, self.lb, self.ub)

    def __call__(self, func):
        particles, velocities, personal_best_positions, personal_best_scores = self.initialize_particles()
        global_best_position = None
        global_best_score = np.inf

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                score = func(particles[i])
                self.evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]

            w = self.w_max - (self.w_max - self.w_min) * (self.evaluations / self.budget)
            scale_factor = np.exp(-self.evaluations / self.budget)

            for i in range(self.population_size):
                velocities[i] = (w * velocities[i] +
                                 self.c1 * np.random.rand() * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * np.random.rand() * (global_best_position - particles[i]) +
                                 self.gamma * (particles[i] - global_best_position))
                velocities[i] = np.clip(velocities[i], -3, 3)  # Limit velocity to prevent overshooting
                particles[i] = particles[i] + velocities[i]
                particles[i] = self.differential_evolution(i, global_best_position, particles)
                particles[i] = np.clip(particles[i], self.lb, self.ub)

            for i in range(self.population_size):
                if self.evaluations < self.budget:
                    score = func(particles[i])
                    self.evaluations += 1
                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = particles[i]
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = particles[i]

        return global_best_position, global_best_score