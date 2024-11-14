import numpy as np

class EnhancedMemoryParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.gamma = 0.2
        self.evaluations = 0

    def chaotic_initialization(self):
        return self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dim)

    def initialize_particles(self):
        particles = self.chaotic_initialization()
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_scores

    def dynamic_adaptive_mutation(self, target, r1, r2):
        scale_factor = np.random.rand()
        mutated = target + self.gamma * scale_factor * (r1 - r2)
        return np.clip(mutated, self.lb, self.ub)

    def social_learning(self, particles, personal_best_positions, global_best_position):
        for i in range(self.population_size):
            if np.random.rand() < 0.5:
                idx = np.random.choice(self.population_size)
                particles[i] = (particles[i] + personal_best_positions[idx] + global_best_position) / 3.0

    def __call__(self, func):
        particles, velocities, personal_best_positions, personal_best_scores = self.initialize_particles()
        global_best_position = None
        global_best_score = np.inf

        while self.evaluations < self.budget:
            self.social_learning(particles, personal_best_positions, global_best_position)
            for i in range(self.population_size):
                score = func(particles[i])
                self.evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = np.copy(particles[i])
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = np.copy(particles[i])

            w = self.w_max - (self.w_max - self.w_min) * (self.evaluations / self.budget)

            for i in range(self.population_size):
                r1, r2 = np.random.choice(self.population_size, 2, replace=False)
                velocities[i] = (w * velocities[i] +
                                 self.c1 * np.random.rand() * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * np.random.rand() * (global_best_position - particles[i]))
                particles[i] = particles[i] + velocities[i]
                particles[i] = self.dynamic_adaptive_mutation(particles[i], personal_best_positions[r1], personal_best_positions[r2])
                particles[i] = np.clip(particles[i], self.lb, self.ub)

            if self.evaluations + self.population_size <= self.budget:
                for i in range(self.population_size):
                    idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)
                    child = (personal_best_positions[idx1] + personal_best_positions[idx2]) / 2
                    score = func(child)
                    self.evaluations += 1
                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = child
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = child

        return global_best_position, global_best_score