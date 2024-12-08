import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40
        self.w = 0.85
        self.w_min = 0.4
        self.c1 = 1.7
        self.c2 = 1.3
        self.temp_init = 100.0
        self.temp_end = 1.0
        self.adaptive_factor = 0.96
        self.bounds = (-5.0, 5.0)
        self.vel_bounds = (-1.0, 1.0)
        self.global_best_pos = np.random.uniform(*self.bounds, self.dim)
        self.global_best_val = float('inf')
        # New: Calculate swarm reduction factor
        self.swarm_reduction = int(self.swarm_size / 10)

    def __call__(self, func):
        np.random.seed(42)
        particles = np.random.uniform(self.bounds[0], self.bounds[1], (self.swarm_size, self.dim))
        velocities = np.random.uniform(self.vel_bounds[0], self.vel_bounds[1], (self.swarm_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.swarm_size, float('inf'))

        evaluations = 0
        temperature = self.temp_init
        swarm_size = self.swarm_size

        while evaluations < self.budget:
            for i in range(swarm_size):
                current_value = func(particles[i])
                evaluations += 1

                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = particles[i]

                if current_value < self.global_best_val:
                    self.global_best_val = current_value
                    self.global_best_pos = particles[i]

            for i in range(swarm_size):
                inertia = self.w * velocities[i]
                cognitive = self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - particles[i])
                social = self.c2 * np.random.rand(self.dim) * (self.global_best_pos - particles[i])
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
            # New: Reduce swarm size as budget depletes
            if evaluations > self.budget / 2 and swarm_size > self.swarm_reduction:
                swarm_size -= self.swarm_reduction

        return self.global_best_pos, self.global_best_val