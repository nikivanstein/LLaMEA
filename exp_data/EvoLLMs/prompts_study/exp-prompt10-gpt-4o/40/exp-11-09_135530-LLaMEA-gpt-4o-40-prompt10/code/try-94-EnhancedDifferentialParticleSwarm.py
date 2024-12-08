import numpy as np

class EnhancedDifferentialParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 50
        self.c1 = 1.2  # Reduced cognitive component
        self.c2 = 1.8  # Increased social component
        self.w_max = 0.9  # Increased inertia
        self.w_min = 0.4  # Increased minimal inertia
        self.adaptive_factor = 0.5  # Differential evolution-inspired factor
        self.evaluations = 0

    def chaotic_initialization(self):
        return self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dim)

    def initialize_particles(self):
        particles = self.chaotic_initialization()
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_scores

    def differential_mutation(self, target, best, r1, r2):
        mutated = target + self.adaptive_factor * (best - target) + self.adaptive_factor * (r1 - r2)
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

            for i in range(self.population_size):
                r1, r2, r3 = np.random.choice(self.population_size, 3, replace=False)
                r1, r2 = personal_best_positions[r1], personal_best_positions[r2]
                velocities[i] = (w * velocities[i] +
                                 self.c1 * np.random.rand() * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * np.random.rand() * (global_best_position - particles[i]))
                particles[i] = particles[i] + velocities[i]
                particles[i] = self.differential_mutation(particles[i], global_best_position, r1, r2)
                particles[i] = np.clip(particles[i], self.lb, self.ub)

            if self.evaluations + self.population_size <= self.budget:
                for i in range(self.population_size):
                    parent1, parent2 = personal_best_positions[np.random.choice(self.population_size, 2, replace=False)]
                    child = self.differential_mutation(parent1, global_best_position, parent1, parent2)
                    score = func(child)
                    self.evaluations += 1
                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = child
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = child

        return global_best_position, global_best_score