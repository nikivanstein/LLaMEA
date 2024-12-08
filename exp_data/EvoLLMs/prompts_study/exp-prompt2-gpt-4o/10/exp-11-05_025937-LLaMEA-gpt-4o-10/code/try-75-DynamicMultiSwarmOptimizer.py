import numpy as np

class DynamicMultiSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 50
        self.num_swarms = 5
        self.inertia_weight = 0.9
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.vel_max = (self.upper_bound - self.lower_bound) * 0.1

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-self.vel_max, self.vel_max, (self.num_particles, self.dim))
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in personal_best])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]
        global_best_value = personal_best_values[global_best_idx]

        function_evaluations = self.num_particles

        while function_evaluations < self.budget:
            inertia = self.inertia_weight * (1 - function_evaluations / self.budget) * 0.95
            self.cognitive_coef = 1.5 + 0.5 * np.sin(2 * np.pi * function_evaluations / self.budget)  # Adaptive learning rate
            self.social_coef = 1.5 + 0.5 * np.cos(2 * np.pi * function_evaluations / self.budget)  # Adaptive learning rate

            for i in range(self.num_particles):
                velocities[i] = (inertia * velocities[i] +
                                 self.cognitive_coef * np.random.rand(self.dim) * (personal_best[i] - particles[i]) +
                                 self.social_coef * np.random.rand(self.dim) * (global_best - particles[i]))
                velocities[i] = np.clip(velocities[i], -self.vel_max, self.vel_max)
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

                value = func(particles[i])
                function_evaluations += 1
                if value < personal_best_values[i]:
                    personal_best[i] = particles[i]
                    personal_best_values[i] = value
                    if value < global_best_value:
                        global_best = particles[i]
                        global_best_value = value

                if function_evaluations >= self.budget:
                    break

                if function_evaluations % (self.budget // 5) == 0:
                    velocities[i] = np.random.uniform(-self.vel_max, self.vel_max, self.dim) * np.random.choice([-1, 1])  # Velocity perturbation

            # Swarm communication: share information with neighboring particles
            if function_evaluations % (self.budget // 10) == 0:
                for j in range(self.num_particles):
                    neighbor_idx = (j + 1) % self.num_particles
                    neighbor_value = func(particles[neighbor_idx])
                    if neighbor_value < personal_best_values[j]:
                        personal_best[j] = particles[neighbor_idx]
                        personal_best_values[j] = neighbor_value

        return global_best