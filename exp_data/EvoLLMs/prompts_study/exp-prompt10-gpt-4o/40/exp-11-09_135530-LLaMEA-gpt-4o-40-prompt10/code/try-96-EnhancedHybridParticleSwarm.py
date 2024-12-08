import numpy as np

class EnhancedHybridParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 50
        self.c1_base = 2.0
        self.c2_base = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.gamma = 0.15  # Memory-based velocity adjustment rate
        self.evaluations = 0

    def chaotic_initialization(self):
        return self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dim)

    def initialize_particles(self):
        particles = self.chaotic_initialization()
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_scores

    def adaptive_mutation(self, target, best, scale_factor):
        direction = np.random.uniform(-1, 1, self.dim)
        mutated = target + scale_factor * (best - target + direction)
        return np.clip(mutated, self.lb, self.ub)

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
            c1 = self.c1_base * (1 - self.evaluations / self.budget)
            c2 = self.c2_base * (self.evaluations / self.budget)
            scale_factor = np.exp(-self.evaluations / self.budget)

            for i in range(self.population_size):
                r1, r2 = np.random.choice(self.population_size, 2, replace=False)
                velocities[i] = (w * velocities[i] +
                                 c1 * np.random.rand() * (personal_best_positions[i] - particles[i]) +
                                 c2 * np.random.rand() * (global_best_position - particles[i]) +
                                 self.gamma * (particles[r1] - particles[r2]))
                particles[i] = particles[i] + velocities[i]
                particles[i] = self.adaptive_mutation(particles[i], global_best_position, scale_factor)
                particles[i] = np.clip(particles[i], self.lb, self.ub)

        return global_best_position, global_best_score