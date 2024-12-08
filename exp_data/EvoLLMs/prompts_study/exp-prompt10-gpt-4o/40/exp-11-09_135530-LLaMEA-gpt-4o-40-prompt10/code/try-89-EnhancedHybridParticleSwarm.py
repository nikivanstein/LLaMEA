import numpy as np

class EnhancedHybridParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 50
        self.c1_initial = 2.5
        self.c2_initial = 0.5
        self.c1_final = 0.5
        self.c2_final = 2.5
        self.w_max = 0.9
        self.w_min = 0.4
        self.alpha = 0.3
        self.gamma = 0.05  # Reduced memory-based velocity adjustment rate
        self.evaluations = 0

    def chaotic_initialization(self):
        return self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dim)

    def initialize_particles(self):
        particles = self.chaotic_initialization()
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_scores

    def multi_strategy_mutation(self, target, best, r1, r2, scale_factor):
        if np.random.rand() < 0.5:
            mutated = target + self.alpha * (best - target) + scale_factor * (r1 - r2)
        else:
            perturbed = target + np.random.normal(0, 0.1, self.dim)
            mutated = self.alpha * perturbed + (1 - self.alpha) * target
        return np.clip(mutated, self.lb, self.ub)

    def adaptive_topology(self, global_best_position, particles):
        avg_position = np.mean(particles, axis=0)
        if np.random.rand() < 0.3:
            influence_point = global_best_position
        else:
            influence_point = avg_position
        return influence_point

    def __call__(self, func):
        particles, velocities, personal_best_positions, personal_best_scores = self.initialize_particles()
        global_best_position = None
        global_best_score = np.inf

        while self.evaluations < self.budget:
            c1 = self.c1_initial - (self.c1_initial - self.c1_final) * (self.evaluations / self.budget)
            c2 = self.c2_initial + (self.c2_final - self.c2_initial) * (self.evaluations / self.budget)

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
                r1, r2, r3 = np.random.choice(self.population_size, 3, replace=False)
                r1, r2 = personal_best_positions[r1], personal_best_positions[r2]
                velocities[i] = (w * velocities[i] +
                                 c1 * np.random.rand() * (personal_best_positions[i] - particles[i]) +
                                 c2 * np.random.rand() * (global_best_position - particles[i]) +
                                 self.gamma * (particles[i] - global_best_position))
                particles[i] = particles[i] + velocities[i]
                particles[i] = self.multi_strategy_mutation(particles[i], global_best_position, r1, r2, scale_factor)
                particles[i] = np.clip(particles[i], self.lb, self.ub)

            influence_point = self.adaptive_topology(global_best_position, particles)

            if self.evaluations + self.population_size <= self.budget:
                for i in range(self.population_size):
                    if np.random.rand() < 0.5:
                        parent1, parent2 = personal_best_positions[np.random.choice(self.population_size, 2, replace=False)]
                    else:
                        parent1, parent2 = global_best_position, influence_point
                    child = (parent1 + parent2) / 2
                    score = func(child)
                    self.evaluations += 1
                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = child
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = child

        return global_best_position, global_best_score