import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        swarm_size = 20
        max_iter = self.budget // swarm_size
        lb = -5.0
        ub = 5.0

        def random_vector():
            return lb + (ub - lb) * np.random.rand(self.dim)

        best_pos = random_vector()
        best_val = func(best_pos)

        swarm = [random_vector() for _ in range(swarm_size)]
        swarm_vals = [func(p) for p in swarm]
        for _ in range(max_iter):
            for i in range(swarm_size):
                r1, r2 = np.random.choice(swarm_size, 2, replace=False)
                vi = swarm[i] + 0.5 * (swarm[r1] - swarm[r2])
                vi = np.clip(vi, lb, ub)
                fi = func(vi)
                if fi < swarm_vals[i]:
                    swarm[i] = vi
                    swarm_vals[i] = fi
                    if fi < best_val:
                        best_val = fi
                        best_pos = vi
        return best_val