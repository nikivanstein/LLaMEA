import numpy as np

class EnhancedParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20 * dim
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.0
        self.neighborhood_size = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def _initialize_swarm(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))

    def _get_fitness(self, swarm, func):
        return np.array([func(individual) for individual in swarm])

    def _update_velocity(self, swarm, velocities, personal_best, global_best):
        inertia_weight = self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * iter_count / max_iter
        for i in range(self.swarm_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            personal_delta = self.cognitive_coeff * r1 * (personal_best[i] - swarm[i])
            global_delta = self.social_coeff * r2 * (global_best - swarm[i])
            velocities[i] = inertia_weight * velocities[i] + personal_delta + global_delta

    def _update_position(self, swarm, velocities):
        return np.clip(swarm + velocities, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        swarm = self._initialize_swarm()
        velocities = np.zeros_like(swarm)
        personal_best = swarm.copy()
        global_best = swarm[np.argmin(self._get_fitness(swarm, func))]
        iter_count = 0
        max_iter = self.budget // self.swarm_size

        while iter_count < max_iter:
            self._update_velocity(swarm, velocities, personal_best, global_best)
            swarm = self._update_position(swarm, velocities)

            fitness_values = self._get_fitness(swarm, func)
            personal_best[fitness_values < self._get_fitness(personal_best, func)] = swarm[fitness_values < self._get_fitness(personal_best, func)]
            global_best = swarm[np.argmin(fitness_values)]

            iter_count += 1

        return global_best