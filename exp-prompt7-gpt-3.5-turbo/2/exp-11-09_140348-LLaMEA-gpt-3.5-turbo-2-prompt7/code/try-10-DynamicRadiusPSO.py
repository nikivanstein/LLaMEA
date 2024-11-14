class DynamicRadiusPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def within_bounds(x):
            return np.clip(x, -5.0, 5.0)

        def local_search(current, global_best, radius):
            new_point = current + np.random.uniform(-radius, radius, size=current.shape)
            new_point = within_bounds(new_point)
            if func(new_point) < func(current):
                return new_point
            return current

        swarm_size = 20
        inertia_weight = 0.5
        cognitive_weight = social_weight = 1.0

        swarm = np.random.uniform(-5.0, 5.0, size=(swarm_size, self.dim))
        velocities = np.zeros((swarm_size, self.dim))
        personal_bests = swarm.copy()
        global_best = swarm[np.argmin([func(p) for p in swarm])]

        for t in range(1, self.budget + 1):
            inertia_weight = 0.4 + 0.1 * (1 - t / self.budget)  # Adaptive inertia weight

            for i in range(swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (inertia_weight * velocities[i] +
                                 cognitive_weight * r1 * (personal_bests[i] - swarm[i]) +
                                 social_weight * r2 * (global_best - swarm[i]))
                swarm[i] = within_bounds(swarm[i] + velocities[i])
                swarm[i] = local_search(swarm[i], global_best, np.linalg.norm(global_best - swarm[i]))

                if func(swarm[i]) < func(personal_bests[i]):
                    personal_bests[i] = swarm[i]
                    if func(swarm[i]) < func(global_best):
                        global_best = swarm[i]

        return global_best