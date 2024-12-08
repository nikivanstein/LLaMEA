import numpy as np

class EnhancedParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_decay = 0.99
        self.min_inertia = 0.1
        self.evaluations = 0

    def chaotic_initialization(self):
        return self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dim)

    def initialize_particles(self):
        particles = self.chaotic_initialization()
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_scores

    def exponential_inertia(self, w_max, evals):
        return max(self.min_inertia, w_max * self.inertia_decay ** evals)

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

            w = self.exponential_inertia(self.inertia_decay, self.evaluations)
            for i in range(self.population_size):
                velocities[i] = (w * velocities[i] +
                                 self.c1 * np.random.rand() * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * np.random.rand() * (global_best_position - particles[i]))
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], self.lb, self.ub)

        return global_best_position, global_best_score