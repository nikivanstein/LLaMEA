import numpy as np
from scipy.stats.qmc import Sobol

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
        # Use Sobol sequences for initializing particles
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        particles = sobol_sampler.random_base2(m=int(np.log2(self.num_particles))) * (self.upper_bound - self.lower_bound) + self.lower_bound
        velocities = np.random.uniform(-self.vel_max, self.vel_max, (self.num_particles, self.dim))
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in personal_best])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]
        global_best_value = personal_best_values[global_best_idx]

        function_evaluations = self.num_particles

        while function_evaluations < self.budget:
            inertia = self.inertia_weight * (1 - function_evaluations / self.budget) * 0.95
            cognitive = self.cognitive_coef * (function_evaluations / self.budget)
            social = self.social_coef * (1 - function_evaluations / self.budget)

            for i in range(self.num_particles):
                velocities[i] = (inertia * velocities[i] +
                                 cognitive * np.random.rand(self.dim) * (personal_best[i] - particles[i]) +
                                 social * np.random.rand(self.dim) * (global_best - particles[i]))
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
                    velocities[i] = np.random.uniform(-self.vel_max, self.vel_max, self.dim)

        return global_best