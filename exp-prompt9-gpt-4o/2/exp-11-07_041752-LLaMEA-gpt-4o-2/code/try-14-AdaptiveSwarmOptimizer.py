import numpy as np

class AdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest_position = None
        self.gbest_score = float('inf')
        self.evaluations = 0
        self.inertia_weight = 0.72  # Slightly increased initial inertia
        self.cognitive_coeff = 1.55  # Slightly adjusted cognitive coefficient
        self.social_coeff = 1.45  # Slightly adjusted social coefficient

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                # Evaluate fitness
                fitness = func(self.positions[i])
                self.evaluations += 1

                # Update personal best
                if fitness < self.pbest_scores[i]:
                    self.pbest_scores[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()

                # Update global best
                if fitness < self.gbest_score:
                    self.gbest_score = fitness
                    self.gbest_position = self.positions[i].copy()

            # Adaptive parameters based on a random phase selector
            if np.random.rand() < 0.48:  # Adjusted threshold for phase selection
                self.inertia_weight = 0.88  # Slightly adjusted for more exploration
                self.cognitive_coeff = 1.9
                self.social_coeff = 1.1
            else:
                self.inertia_weight = 0.42  # Slightly adjusted for more exploitation
                self.cognitive_coeff = 1.1
                self.social_coeff = 1.9

            # Update velocities and positions
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_coeff * r1 * (self.pbest_positions[i] - self.positions[i])
                social_velocity = self.social_coeff * r2 * (self.gbest_position - self.positions[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)

                # Update position
                self.positions[i] += self.velocities[i]

                # Ensure position is within bounds
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.gbest_position, self.gbest_score