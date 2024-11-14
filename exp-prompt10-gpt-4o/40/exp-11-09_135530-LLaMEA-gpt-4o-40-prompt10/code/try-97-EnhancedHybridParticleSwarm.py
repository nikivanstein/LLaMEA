import numpy as np

class EnhancedHybridParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_max = 0.9
        self.w_min = 0.4
        self.alpha = 0.5
        self.beta = 0.7
        self.gamma = 0.2  # Memory-based velocity adjustment rate
        self.evaluations = 0
        self.elite_fraction = 0.1

    def chaotic_initialization(self):
        return self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dim)

    def initialize_particles(self):
        particles = self.chaotic_initialization()
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_scores

    def differential_crossover(self, parent1, parent2, parent3):
        mutant = parent1 + self.beta * (parent2 - parent3)
        return np.clip(mutant, self.lb, self.ub)

    def adaptive_mutation(self, target, best, r1, r2, scale_factor):
        mutated = target + self.beta * (best - target) + scale_factor * (r1 - r2)
        return np.clip(mutated, self.lb, self.ub)

    def elite_selection(self, scores, particles):
        elite_size = int(self.elite_fraction * self.population_size)
        elite_indices = np.argsort(scores)[:elite_size]
        return particles[elite_indices]

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
                elite = self.elite_selection(personal_best_scores, personal_best_positions)
                r1, r2, r3 = np.random.choice(elite.shape[0], 3, replace=False)
                r1, r2, r3 = elite[r1], elite[r2], elite[r3]
                velocities[i] = (w * velocities[i] +
                                 self.c1 * np.random.rand() * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * np.random.rand() * (global_best_position - particles[i]) +
                                 self.gamma * (particles[i] - global_best_position))
                particles[i] = particles[i] + velocities[i]
                particles[i] = self.adaptive_mutation(particles[i], global_best_position, r1, r2, scale_factor)
                particles[i] = np.clip(particles[i], self.lb, self.ub)

            if self.evaluations + self.population_size <= self.budget:
                for i in range(self.population_size):
                    parent1, parent2, parent3 = elite[np.random.choice(elite.shape[0], 3, replace=False)]
                    child = self.differential_crossover(parent1, parent2, parent3)
                    score = func(child)
                    self.evaluations += 1
                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = child
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = child

        return global_best_position, global_best_score