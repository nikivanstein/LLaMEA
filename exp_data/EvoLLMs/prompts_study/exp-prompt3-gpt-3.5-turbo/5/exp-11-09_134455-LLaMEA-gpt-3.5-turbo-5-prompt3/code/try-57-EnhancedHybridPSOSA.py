import numpy as np
from scipy.optimize import minimize

class EnhancedHybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 20
        self.max_iter = budget // self.num_particles
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_min = 0.4
        self.w_max = 0.9
        self.T_init = 1000.0
        self.T_min = 1e-8
        self.diversity_factor = 0.1  # New parameter for diversity maintenance
        self.neighborhood_factor = 0.2

    def __call__(self, func):
        def pso_sa_optimize():
            positions = np.zeros((self.num_particles, self.dim))
            for d in range(self.dim):
                step_size = 10.0 / self.num_particles
                for p in range(self.num_particles):
                    positions[p][d] = np.random.uniform(step_size * p, step_size * (p + 1))

            velocities = np.zeros((self.num_particles, self.dim))
            pbest_positions = np.copy(positions)
            pbest_values = np.array([func(p) for p in pbest_positions])
            gbest_position = pbest_positions[np.argmin(pbest_values)]
            gbest_value = np.min(pbest_values)
            T = self.T_init

            for _ in range(self.max_iter):
                for i in range(self.num_particles):
                    w = self.w_min + (_ / self.max_iter) * (self.w_max - self.w_min)
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[i] = w * velocities[i] + self.c1 * r1 * (pbest_positions[i] - positions[i]) + self.c2 * r2 * (gbest_position - positions[i])
                    
                    # Dynamic neighborhood search
                    neighborhood_size = int(self.num_particles * self.neighborhood_factor)
                    neighborhood_indices = np.random.choice(self.num_particles, neighborhood_size, replace=False)
                    for j in neighborhood_indices:
                        velocities[i] += self.c1 * r1 * (pbest_positions[j] - positions[i]) + self.c2 * r2 * (gbest_position - positions[i])

                    positions[i] = np.clip(positions[i] + velocities[i], -5.0, 5.0)

                    res = minimize(func, positions[i], method='Nelder-Mead')
                    if res.fun < pbest_values[i]:
                        pbest_positions[i] = res.x
                        pbest_values[i] = res.fun

                    if res.fun < gbest_value:
                        gbest_position = res.x
                        gbest_value = res.fun

                T *= 0.99 if T > self.T_min else 1.0

                # Introducing diversity maintenance mechanism
                random_particle = np.random.randint(self.num_particles)
                random_position = np.random.uniform(-5.0, 5.0, size=self.dim)
                if func(random_position) < pbest_values[random_particle]:
                    positions[random_particle] = random_position
                    pbest_positions[random_particle] = random_position
                    pbest_values[random_particle] = func(random_position)

            return gbest_value

        return pso_sa_optimize()