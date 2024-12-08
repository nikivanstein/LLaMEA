import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 15
        self.inertia_weight = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = 0.5
        self.CR = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.vel_max = (self.upper_bound - self.lower_bound) * 0.1
        self.vel_min = -self.vel_max

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(self.vel_min, self.vel_max, (self.num_particles, self.dim))
        p_best = particles.copy()
        p_best_values = np.array([func(p) for p in particles])
        g_best = p_best[np.argmin(p_best_values)]
        g_best_value = np.min(p_best_values)
        eval_count = self.num_particles

        while eval_count < self.budget:
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2)
                self.inertia_weight *= 0.99  # Adaptive inertia
                velocities[i] = (self.inertia_weight * velocities[i] 
                                 + self.c1 * r1 * (p_best[i] - particles[i]) 
                                 + self.c2 * r2 * (g_best - particles[i]))
                velocities[i] = np.clip(velocities[i], self.vel_min, self.vel_max)
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

                value = func(particles[i])
                eval_count += 1

                if value < p_best_values[i]:
                    p_best[i] = particles[i].copy()
                    p_best_values[i] = value

                if value < g_best_value:
                    g_best = particles[i].copy()
                    g_best_value = value

                if eval_count >= self.budget:
                    break

            if eval_count < self.budget:
                for i in range(self.num_particles):
                    indices = [idx for idx in range(self.num_particles) if idx != i]
                    a, b, c = np.random.choice(indices, 3, replace=False)

                    mutant = p_best[a] + self.F * (p_best[b] - p_best[c])
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                    trial = np.where(np.random.rand(self.dim) < self.CR, mutant, particles[i])
                    trial_value = func(trial)
                    eval_count += 1

                    if trial_value < p_best_values[i]:
                        p_best[i] = trial
                        p_best_values[i] = trial_value

                        if trial_value < g_best_value:
                            g_best = trial
                            g_best_value = trial_value

                    if eval_count >= self.budget:
                        break

        return g_best, g_best_value