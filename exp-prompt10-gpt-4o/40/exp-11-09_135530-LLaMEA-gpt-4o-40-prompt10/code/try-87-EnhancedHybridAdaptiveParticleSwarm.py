import numpy as np

class EnhancedHybridAdaptiveParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 50
        self.c1 = 2.0  # Increased cognitive component
        self.c2 = 2.0  # Increased social component
        self.w_max = 0.9  # Increased inertia weight for exploration
        self.w_min = 0.4  # Increased minimum inertia weight
        self.alpha = 0.6  # Increased crossover probability
        self.beta = 0.8  # Increased mutation impact
        self.gamma = 0.15  # Increased memory-based velocity adjustment rate
        self.evaluations = 0

    def chaotic_initialization(self):
        return self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dim)

    def initialize_particles(self):
        particles = self.chaotic_initialization()
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_scores

    def dynamic_crossover(self, parent1, parent2):
        mix_ratio = np.random.rand(self.dim) < self.alpha
        child = np.where(mix_ratio, parent1, parent2)
        return np.clip(child, self.lb, self.ub)

    def adaptive_mutation(self, target, best, r1, r2, scale_factor):
        mutated = target + self.beta * (best - target) + scale_factor * (r1 - r2)
        return np.clip(mutated, self.lb, self.ub)

    def __call__(self, func):
        particles, velocities, personal_best_positions, personal_best_scores = self.initialize_particles()
        global_best_position = None
        global_best_score = np.inf
        neighborhood_size = max(1, self.population_size // 5)

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
                neighborhood = np.random.choice(self.population_size, neighborhood_size, replace=False)
                local_best_position = personal_best_positions[neighborhood[np.argmin(personal_best_scores[neighborhood])]]
                velocities[i] = (w * velocities[i] +
                                 self.c1 * np.random.rand() * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * np.random.rand() * (global_best_position - particles[i]) +
                                 self.gamma * (local_best_position - particles[i]))
                particles[i] = particles[i] + velocities[i]
                particles[i] = self.adaptive_mutation(particles[i], global_best_position, particles[i], local_best_position, scale_factor)
                particles[i] = np.clip(particles[i], self.lb, self.ub)

            if self.evaluations + self.population_size <= self.budget:
                for i in range(self.population_size):
                    parent1, parent2 = personal_best_positions[np.random.choice(self.population_size, 2, replace=False)]
                    child = self.dynamic_crossover(parent1, parent2)
                    score = func(child)
                    self.evaluations += 1
                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = child
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = child

        return global_best_position, global_best_score