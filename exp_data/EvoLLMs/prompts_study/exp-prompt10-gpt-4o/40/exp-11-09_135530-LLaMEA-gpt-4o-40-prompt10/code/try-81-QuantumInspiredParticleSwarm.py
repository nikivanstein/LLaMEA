import numpy as np

class QuantumInspiredParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 50
        self.c1_initial = 2.5
        self.c2_initial = 0.5
        self.w_max = 0.9
        self.w_min = 0.4
        self.evaluations = 0

    def quantum_initialization(self):
        center = (self.lb + self.ub) / 2
        spread = (self.ub - self.lb) / 2
        return center + spread * (2 * np.random.rand(self.population_size, self.dim) - 1)

    def initialize_particles(self):
        particles = self.quantum_initialization()
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_scores

    def dynamic_parameter_adjustment(self):
        progress = self.evaluations / self.budget
        c1 = self.c1_initial - (self.c1_initial - self.c2_initial) * progress
        c2 = self.c2_initial + (self.c1_initial - self.c2_initial) * progress
        return c1, c2

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
            c1, c2 = self.dynamic_parameter_adjustment()

            for i in range(self.population_size):
                velocities[i] = (w * velocities[i] +
                                 c1 * np.random.rand() * (personal_best_positions[i] - particles[i]) +
                                 c2 * np.random.rand() * (global_best_position - particles[i]))
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], self.lb, self.ub)

        return global_best_position, global_best_score