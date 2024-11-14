import numpy as np

class EnhancedParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 50
        self.c1 = 1.4
        self.c2 = 1.8
        self.w_max = 0.9
        self.w_min = 0.4
        self.evaluations = 0

    def initialize_particles(self):
        particles = self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dim)
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_scores

    def dynamic_learning_factor(self):
        return self.c1 + (self.evaluations / self.budget) * (self.c2 - self.c1)

    def stochastic_velocity_control(self, velocity, particle, personal_best, global_best):
        random_factor = np.random.rand(self.dim)
        inertia = (self.w_max - (self.w_max - self.w_min) * (self.evaluations / self.budget))
        cognitive = self.dynamic_learning_factor() * random_factor * (personal_best - particle)
        social = self.dynamic_learning_factor() * random_factor * (global_best - particle)
        new_velocity = inertia * velocity + cognitive + social
        return new_velocity

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

            for i in range(self.population_size):
                velocities[i] = self.stochastic_velocity_control(velocities[i], particles[i],
                                                                 personal_best_positions[i], global_best_position)
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], self.lb, self.ub)

        return global_best_position, global_best_score