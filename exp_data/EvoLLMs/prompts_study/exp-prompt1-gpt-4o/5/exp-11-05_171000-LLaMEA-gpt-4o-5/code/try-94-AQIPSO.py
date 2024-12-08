import numpy as np

class AQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(2 * np.sqrt(dim))
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w_max = 0.9  # max inertia weight
        self.w_min = 0.4  # min inertia weight
        self.best_global_position = None
        self.best_global_value = np.inf
        self.evaluations = 0

    def initialize(self):
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.best_personal_positions = np.copy(self.positions)
        self.best_personal_values = np.full(self.population_size, np.inf)

    def quantum_update(self, position, global_best):
        phi = np.random.uniform(0, 1, self.dim)
        delta = np.abs(position - global_best)
        new_position = global_best + (-1)**np.random.randint(2, size=self.dim) * delta * np.log(1/phi)
        return np.clip(new_position, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        self.initialize()
        while self.evaluations < self.budget:
            self.w = self.w_max - (self.w_max - self.w_min) * (self.evaluations / self.budget)  # Adaptive inertia
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Evaluate current position
                value = func(self.positions[i])
                self.evaluations += 1

                # Update personal best
                if value < self.best_personal_values[i]:
                    self.best_personal_values[i] = value
                    self.best_personal_positions[i] = self.positions[i]

                # Update global best
                if value < self.best_global_value:
                    self.best_global_value = value
                    self.best_global_position = self.positions[i]

            if self.evaluations >= self.budget:
                break

            # Update velocity and positions
            for i in range(self.population_size):
                r1 = np.random.uniform(0, 1, self.dim)
                r2 = np.random.uniform(0, 1, self.dim)

                cognitive_velocity = self.c1 * r1 * (self.best_personal_positions[i] - self.positions[i])
                social_velocity = self.c2 * r2 * (self.best_global_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_velocity + social_velocity

                # Quantum-inspired update
                if np.random.rand() < 0.5:  # Probability threshold for quantum update
                    self.positions[i] = self.quantum_update(self.positions[i], self.best_global_position)
                else:
                    self.positions[i] += self.velocities[i]
                    self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

        return self.best_global_value, self.best_global_position