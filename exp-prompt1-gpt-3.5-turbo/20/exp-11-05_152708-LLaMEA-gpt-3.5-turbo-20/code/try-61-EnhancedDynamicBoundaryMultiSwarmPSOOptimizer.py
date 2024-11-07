import numpy as np

class EnhancedDynamicBoundaryMultiSwarmPSOOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_swarms = 5

    def __call__(self, func):
        def initialize_swarm():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def update_swarm_position(swarm, swarm_best, global_best):
            inertia_weight = 0.5
            cognitive_weight = 1.5
            social_weight = 1.5
            velocity = np.zeros(self.dim)
            for i in range(self.dim):
                cognitive_component = cognitive_weight * np.random.rand() * (swarm_best[i] - swarm[i])
                social_component = social_weight * np.random.rand() * (global_best[i] - swarm[i])
                velocity[i] = inertia_weight * velocity[i] + cognitive_component + social_component
                swarm[i] = np.clip(swarm[i] + velocity[i], self.lower_bound, self.upper_bound)

        swarm_size = 10
        swarms = [np.array([initialize_swarm() for _ in range(swarm_size)]) for _ in range(self.num_swarms)]
        swarm_bests = [swarm[np.argmin([func(p) for p in swarm])] for swarm in swarms]
        global_best = min(swarm_bests, key=func)

        for _ in range(self.budget - self.num_swarms * swarm_size):
            for i in range(self.num_swarms):
                for j in range(swarm_size):
                    update_swarm_position(swarms[i][j], swarm_bests[i], global_best)
                    if func(swarms[i][j]) < func(swarm_bests[i]):
                        swarm_bests[i] = swarms[i][j]
                        if func(swarms[i][j]) < func(global_best):
                            global_best = swarms[i][j]

        return global_best