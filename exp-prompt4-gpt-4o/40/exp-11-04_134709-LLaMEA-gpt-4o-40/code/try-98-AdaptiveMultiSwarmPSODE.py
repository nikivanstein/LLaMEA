import numpy as np

class AdaptiveMultiSwarmPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_swarms = 3  # Introducing multiple swarms
        self.num_particles = 5  # Adjusted per swarm
        self.inertia_weight = 0.7
        self.c1 = 2.0
        self.c2 = 2.0
        self.F = 0.8
        self.CR = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.vel_max = (self.upper_bound - self.lower_bound) * 0.1
        self.alpha = 0.5  # New parameter for weighted mutation

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, 
                                      (self.num_swarms, self.num_particles, self.dim))
        velocities = np.random.uniform(-self.vel_max, self.vel_max, 
                                       (self.num_swarms, self.num_particles, self.dim))
        p_best = particles.copy()
        p_best_values = np.array([[func(p) for p in swarm] for swarm in particles])
        g_best = p_best[np.unravel_index(np.argmin(p_best_values), p_best_values.shape)]
        g_best_value = np.min(p_best_values)

        eval_count = self.num_swarms * self.num_particles

        while eval_count < self.budget:
            for swarm_idx in range(self.num_swarms):
                for i in range(self.num_particles):
                    self.inertia_weight = 0.4 + 0.3 * np.random.random()  # Adaptive inertia weight
                    r1, r2 = np.random.rand(2)
                    velocities[swarm_idx][i] = (self.inertia_weight * velocities[swarm_idx][i] 
                                                + self.c1 * r1 * (p_best[swarm_idx][i] - particles[swarm_idx][i]) 
                                                + self.c2 * r2 * (g_best - particles[swarm_idx][i]))
                    velocities[swarm_idx][i] = np.clip(velocities[swarm_idx][i], -self.vel_max, self.vel_max)
                    particles[swarm_idx][i] += velocities[swarm_idx][i]
                    particles[swarm_idx][i] = np.clip(particles[swarm_idx][i], self.lower_bound, self.upper_bound)

                    value = func(particles[swarm_idx][i])
                    eval_count += 1

                    if value < p_best_values[swarm_idx][i]:
                        p_best[swarm_idx][i] = particles[swarm_idx][i].copy()
                        p_best_values[swarm_idx][i] = value

                        if value < g_best_value:
                            g_best = particles[swarm_idx][i].copy()
                            g_best_value = value

                    if eval_count >= self.budget:
                        break

            if eval_count < self.budget:
                for swarm_idx in range(self.num_swarms):
                    for i in range(self.num_particles):
                        indices = [idx for idx in range(self.num_particles) if idx != i]
                        a, b, c = np.random.choice(indices, 3, replace=False)

                        mutant = p_best[swarm_idx][a] + self.F * (p_best[swarm_idx][b] - p_best[swarm_idx][c])
                        hybrid_mutant = self.alpha * mutant + (1 - self.alpha) * np.random.uniform(self.lower_bound, self.upper_bound, self.dim)  # Hybrid mutation
                        hybrid_mutant = np.clip(hybrid_mutant, self.lower_bound, self.upper_bound)

                        trial = np.where(np.random.rand(self.dim) < self.CR, hybrid_mutant, particles[swarm_idx][i])
                        
                        trial_value = func(trial)
                        eval_count += 1

                        if trial_value < p_best_values[swarm_idx][i]:
                            p_best[swarm_idx][i] = trial
                            p_best_values[swarm_idx][i] = trial_value

                            if trial_value < g_best_value:
                                g_best = trial
                                g_best_value = trial_value

                        if eval_count >= self.budget:
                            break

        return g_best, g_best_value