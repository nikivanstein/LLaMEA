import numpy as np

class Quantum_Enhanced_Dual_Swarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Swarm parameters
        self.num_particles = 50
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_coeff = 1.6
        self.social_coeff = 1.8

        # Dual-swarm setup
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def levy_flight(self, size):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        return u / abs(v) ** (1 / beta)

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            inertia_weight = self.inertia_weight_final + (self.inertia_weight_initial - self.inertia_weight_final) * ((self.budget - evals) / self.budget)
            
            scores = np.apply_along_axis(func, 1, self.positions)
            evals += self.num_particles

            for i in range(self.num_particles):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]

                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]

            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            self.velocities = inertia_weight * self.velocities + cognitive_component + social_component

            self.positions += self.velocities
            quantum_step = self.levy_flight((self.num_particles, self.dim))
            self.positions += 0.001 * quantum_step * (self.global_best_position - self.positions)
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score