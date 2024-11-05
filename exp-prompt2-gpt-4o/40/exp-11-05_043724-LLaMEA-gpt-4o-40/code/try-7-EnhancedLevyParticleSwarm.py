import numpy as np

class EnhancedLevyParticleSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.inertia_weight = 0.9
        self.inertia_weight_min = 0.4
        self.inertia_weight_max = 0.9
        self.cognitive_component = 2.0
        self.social_component = 2.0
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def levy_flight(self, L):
        sigma = (np.math.gamma(1 + L) * np.sin(np.pi * L / 2) / 
                 (np.math.gamma((1 + L) / 2) * L * 2 ** ((L - 1) / 2))) ** (1 / L)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / abs(v) ** (1 / L)
        return step

    def chaotic_mutation(self, position, chaotic_factor):
        return position + chaotic_factor * np.random.uniform(-1, 1, self.dim)

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Evaluate the fitness of the particle
                fitness = func(self.positions[i])
                self.evaluations += 1

                # Update personal best
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

                # Update global best
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i]

            # Calculate adaptive inertia weight
            inertia_weight = self.inertia_weight_max - ((self.inertia_weight_max - self.inertia_weight_min) * 
                                                        (self.evaluations / self.budget))

            # Update velocities and positions
            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                cognitive_velocity = self.cognitive_component * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_component * r2 * (self.global_best_position - self.positions[i])
                levy_step = self.levy_flight(1.5)
                self.velocities[i] = (inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity + levy_step)
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Perform chaotic mutation for diversification
            if np.random.rand() < 0.1:  # Mutation probability
                for i in range(self.population_size):
                    chaotic_factor = np.random.rand()
                    self.positions[i] = self.chaotic_mutation(self.positions[i], chaotic_factor)

        return self.global_best_score