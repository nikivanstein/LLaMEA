import numpy as np

class DynamicInertiaHybridPSODEImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        max_swarm_size = 20
        min_swarm_size = 5
        lb = -5.0
        ub = 5.0

        def random_vector():
            return lb + (ub - lb) * np.random.rand(self.dim)

        best_pos = random_vector()
        best_val = func(best_pos)

        inertia_weight = 0.5  # Initialize inertia weight factor

        swarm_size = max_swarm_size
        max_iter = self.budget // swarm_size

        swarm = [random_vector() for _ in range(swarm_size)]
        swarm_vals = [func(p) for p in swarm]
        for _ in range(max_iter):
            for i in range(swarm_size):
                r1, r2, r3 = np.random.choice(swarm_size, 3, replace=False)
                vi = swarm[i] + inertia_weight * (swarm[r1] - swarm[r2]) + 0.5 * (swarm[r3] - swarm[i])
                vi = np.clip(vi, lb, ub)
                fi = func(vi)
                if fi < swarm_vals[i]:
                    swarm[i] = vi
                    swarm_vals[i] = fi
                    if fi < best_val:
                        best_val = fi
                        best_pos = vi
            inertia_weight = 0.5 + 0.5 * (1 - (_ / max_iter))  # Update inertia weight adaptively

            # Dynamic adjustment of swarm_size based on population diversity
            diversity = np.std(swarm_vals)
            if diversity < 0.1:
                swarm_size = min(swarm_size * 2, max_swarm_size)
            elif diversity > 0.5:
                swarm_size = max(swarm_size // 2, min_swarm_size)
            max_iter = self.budget // swarm_size

        return best_val