import numpy as np

class EnhancedParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 60  # Increased population for diversity
        self.c1 = 2.0  # Adjusted cognitive component
        self.c2 = 2.0  # Adjusted social component
        self.w_max = 0.9  # Increased inertia weight for exploration
        self.w_min = 0.4  # Adjusted inertia weight for exploitation
        self.alpha = 0.6  # Adjusted crossover probability
        self.beta = 0.8  # Adjusted mutation factor
        self.gamma = 0.05  # Reduced memory-based velocity adjustment rate
        self.evaluations = 0

    def chaotic_initialization(self):
        # Enhanced chaotic initialization with logistic map
        x0 = np.random.rand(self.population_size, self.dim)
        return self.lb + (self.ub - self.lb) * (4 * x0 * (1 - x0))

    def initialize_particles(self):
        particles = self.chaotic_initialization()
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_scores

    def adaptive_crossover(self, parent1, parent2):
        # Tweaked crossover technique to enhance search diversity
        if np.random.rand() < self.alpha:
            cross_point = np.random.randint(1, self.dim)
            child = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
        else:
            child = (np.sin(parent1) + np.cos(parent2)) / 2
        return np.clip(child, self.lb, self.ub)

    def adaptive_mutation(self, target, best, r1, r2, scale_factor):
        # Mutation adjustment for better exploration
        mutated = target + self.beta * (best - target) + scale_factor * (r1 - r2) * np.random.normal(0, 1, self.dim)
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
            scale_factor = np.exp(-self.evaluations / self.budget)

            for i in range(self.population_size):
                r1, r2, r3 = np.random.choice(self.population_size, 3, replace=False)
                r1, r2 = personal_best_positions[r1], personal_best_positions[r2]
                velocities[i] = (w * velocities[i] +
                                 self.c1 * np.random.rand() * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * np.random.rand() * (global_best_position - particles[i]) +
                                 self.gamma * (particles[i] - global_best_position))
                particles[i] = particles[i] + velocities[i]
                particles[i] = self.adaptive_mutation(particles[i], global_best_position, r1, r2, scale_factor)
                particles[i] = np.clip(particles[i], self.lb, self.ub)

            # Elitist selection: keep top-performing particles
            if self.evaluations + self.population_size <= self.budget:
                for i in range(self.population_size):
                    parent1, parent2 = personal_best_positions[np.random.choice(self.population_size, 2, replace=False)]
                    child = self.adaptive_crossover(parent1, parent2)
                    score = func(child)
                    self.evaluations += 1
                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = child
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = child

        return global_best_position, global_best_score