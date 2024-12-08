import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40
        self.w = 0.85  # Slightly adjusted initial inertia weight
        self.w_min = 0.4
        self.c1_min, self.c1_max = 1.5, 2.0  # Variable cognitive coefficient
        self.c2_min, self.c2_max = 1.0, 1.5  # Variable social coefficient
        self.temp_init = 100.0
        self.temp_end = 1.0
        self.adaptive_factor = 0.96  # Adjusted adaptive cooling rate
        self.bounds = (-5.0, 5.0)
        self.vel_bounds = (-1.0, 1.0)
        self.global_best_pos = np.random.uniform(*self.bounds, self.dim)
        self.global_best_val = float('inf')

    def __call__(self, func):
        np.random.seed(42)
        particles = np.random.uniform(self.bounds[0], self.bounds[1], (self.swarm_size, self.dim))
        velocities = np.random.uniform(self.vel_bounds[0], self.vel_bounds[1], (self.swarm_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.swarm_size, float('inf'))

        evaluations = 0
        temperature = self.temp_init

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                current_value = func(particles[i])
                evaluations += 1

                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = particles[i]

                if current_value < self.global_best_val:
                    self.global_best_val = current_value
                    self.global_best_pos = particles[i]

            for i in range(self.swarm_size):
                c1_dynamic = self.c1_min + (self.c1_max - self.c1_min) * evaluations / self.budget
                c2_dynamic = self.c2_min + (self.c2_max - self.c2_min) * (self.budget - evaluations) / self.budget
                inertia = self.w * velocities[i]
                cognitive = c1_dynamic * np.random.rand(self.dim) * (personal_best_positions[i] - particles[i])
                social = c2_dynamic * np.random.rand(self.dim) * (self.global_best_pos - particles[i])
                velocities[i] = inertia + cognitive + social
                velocities[i] = np.clip(velocities[i], *self.vel_bounds)

                proposed_position = particles[i] + velocities[i]
                proposed_position = np.clip(proposed_position, *self.bounds)

                proposed_value = func(proposed_position)
                evaluations += 1

                if proposed_value < current_value or np.random.rand() < np.exp((current_value - proposed_value) / temperature):
                    particles[i] = proposed_position
                    current_value = proposed_value
                    if current_value < personal_best_values[i]:
                        personal_best_values[i] = current_value
                        personal_best_positions[i] = proposed_position

            temperature *= self.adaptive_factor
            self.w = max(self.w_min, self.w * 0.99)

        return self.global_best_pos, self.global_best_val