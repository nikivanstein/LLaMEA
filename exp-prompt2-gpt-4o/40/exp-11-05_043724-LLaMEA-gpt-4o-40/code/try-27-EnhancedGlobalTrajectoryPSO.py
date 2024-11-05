import numpy as np

class EnhancedGlobalTrajectoryPSO:
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
        self.adaptive_learning_rate = 0.01

    def chaotic_trajectory(self, iteration):
        return np.sin(2 * np.pi * iteration / self.budget) * np.cos(5 * np.pi * iteration / self.budget)

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

            # Calculate adaptive inertia weight using chaotic trajectory
            inertia_weight = self.inertia_weight_max - ((self.inertia_weight_max - self.inertia_weight_min) * 
                                                        (self.evaluations / self.budget))
            trajectory_factor = self.chaotic_trajectory(self.evaluations)
            inertia_weight *= (1 + trajectory_factor)

            # Update velocities and positions with adaptive learning rate
            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)
                cognitive_velocity = self.cognitive_component * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_component * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      cognitive_velocity * (1 + self.adaptive_learning_rate) + 
                                      social_velocity * (1 - self.adaptive_learning_rate))
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Perform mutation influenced by chaotic trajectory
            for i in range(self.population_size):
                if np.random.rand() < 0.2:
                    mutation_point = np.random.randint(0, self.dim)
                    self.positions[i][mutation_point] += trajectory_factor * np.random.normal()

        return self.global_best_score