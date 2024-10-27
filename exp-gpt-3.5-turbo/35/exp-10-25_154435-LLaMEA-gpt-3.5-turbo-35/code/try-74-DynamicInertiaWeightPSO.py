import numpy as np

class DynamicInertiaWeightPSO(ParticleSwarmOptimization):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.distance_weight = 0.5

    def __call__(self, func):
        def calculate_distance_matrix(swarm):
            distances = np.linalg.norm(swarm[:, np.newaxis] - swarm, axis=2)
            np.fill_diagonal(distances, np.inf)
            return distances

        def update_inertia_weight(swarm, distances):
            inertia_weights = np.ones(self.swarm_size) * self.inertia_weight
            for i, particle in enumerate(swarm):
                neighbors = np.argsort(distances[i])[:3]  # Consider the 3 nearest particles
                fitness_difference = np.abs(func(particle) - func(swarm[neighbors]))
                inertia_weights[i] = np.mean(fitness_difference) * self.distance_weight
            return inertia_weights

        swarm, velocities = initialize_swarm(self.swarm_size, self.dim, self.lb, self.ub)
        p_best = swarm.copy()
        g_best = p_best[np.argmin([func(p) for p in p_best])
        for _ in range(self.budget // self.swarm_size):
            distances = calculate_distance_matrix(swarm)
            inertia_weights = update_inertia_weight(swarm, distances)
            for i in range(self.swarm_size):
                self.inertia_weight = inertia_weights[i]
                velocities[i] = update_velocity(swarm[i], velocities[i], p_best[i], g_best)
                swarm[i] = update_position(swarm[i], velocities[i], self.lb, self.ub)
                if func(swarm[i]) < func(p_best[i]):
                    p_best[i] = swarm[i]
            g_best = p_best[np.argmin([func(p) for p in p_best])]
        return g_best