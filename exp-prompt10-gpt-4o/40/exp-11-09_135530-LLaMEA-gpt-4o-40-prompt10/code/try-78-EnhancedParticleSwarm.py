import numpy as np

class EnhancedParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 50
        self.c1 = 2.0  # Increased cognitive parameter
        self.c2 = 2.0  # Increased social parameter
        self.w_max = 0.9
        self.w_min = 0.4
        self.elite_fraction = 0.2  # Fraction of elite particles
        self.evaluations = 0

    def chaotic_initialization(self):
        return self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dim)

    def initialize_particles(self):
        particles = self.chaotic_initialization()
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_scores

    def elite_selection(self, scores):
        elite_count = int(self.elite_fraction * self.population_size)
        elite_indices = np.argsort(scores)[:elite_count]
        return elite_indices

    def adaptive_tuning(self, iteration):
        w = self.w_min + (self.w_max - self.w_min) * (1 - iteration / self.budget)
        c1 = self.c1 * (1 - iteration / self.budget)
        c2 = self.c2 * (iteration / self.budget)
        return w, c1, c2

    def __call__(self, func):
        particles, velocities, personal_best_positions, personal_best_scores = self.initialize_particles()
        global_best_position = None
        global_best_score = np.inf

        while self.evaluations < self.budget:
            current_iteration = self.evaluations // self.population_size
            w, c1, c2 = self.adaptive_tuning(current_iteration)

            for i in range(self.population_size):
                score = func(particles[i])
                self.evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]

            for i in range(self.population_size):
                if i in self.elite_selection(personal_best_scores):
                    # Elite particles randomly shift slightly to explore
                    velocities[i] += np.random.normal(0, 0.1, self.dim)
                else:
                    # Regular update
                    velocities[i] = (w * velocities[i] +
                                     c1 * np.random.rand() * (personal_best_positions[i] - particles[i]) +
                                     c2 * np.random.rand() * (global_best_position - particles[i]))
                particles[i] = np.clip(particles[i] + velocities[i], self.lb, self.ub)

        return global_best_position, global_best_score