import numpy as np

class AdaptiveHybridPSOSA(HybridPSOSA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.initial_w = self.w

    def __call__(self, func):
        def adaptive_pso_sa_optimize():
            # Initialize PSO parameters
            positions = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
            velocities = np.zeros((self.num_particles, self.dim))
            pbest_positions = np.copy(positions)
            pbest_values = np.array([func(p) for p in pbest_positions])
            gbest_position = pbest_positions[np.argmin(pbest_values)]
            gbest_value = np.min(pbest_values)
            T = self.T_init

            for _ in range(self.max_iter):
                # Adaptive inertia weight update
                self.w = self.initial_w + 0.4 * (_ / self.max_iter)

                for i in range(self.num_particles):
                    # PSO update
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[i] = self.w * velocities[i] + self.c1 * r1 * (pbest_positions[i] - positions[i]) + self.c2 * r2 * (gbest_position - positions[i])
                    positions[i] = np.clip(positions[i] + velocities[i], -5.0, 5.0)

                    # Simulated Annealing
                    candidate_position = positions[i] + np.random.normal(0, 0.1, size=self.dim)
                    candidate_position = np.clip(candidate_position, -5.0, 5.0)
                    candidate_value = func(candidate_position)

                    if candidate_value < pbest_values[i]:
                        pbest_positions[i] = candidate_position
                        pbest_values[i] = candidate_value

                    if candidate_value < gbest_value:
                        gbest_position = candidate_position
                        gbest_value = candidate_value
                    else:
                        delta = candidate_value - pbest_values[i]
                        if np.exp(-delta / T) > np.random.rand():
                            positions[i] = candidate_position
                            pbest_values[i] = candidate_value

                T *= 0.99 if T > self.T_min else 1.0

            return gbest_value

        return adaptive_pso_sa_optimize()