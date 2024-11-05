import numpy as np

class AdaptiveMomentumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * int(np.sqrt(dim))  # population size
        self.inertia = 0.9  # initial inertia weight
        self.c1 = 1.2  # cognitive component
        self.c2 = 1.5  # social component
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize the swarm
        pos = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        vel = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        p_best_pos = pos.copy()
        p_best_val = np.array([func(ind) for ind in pos])
        g_best_pos = p_best_pos[np.argmin(p_best_val)]
        g_best_val = np.min(p_best_val)

        evaluations = self.pop_size

        while evaluations < self.budget:
            # Dynamically adjust inertia weight
            self.inertia = 0.9 - (0.5 * (evaluations / self.budget))
            
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            vel = (self.inertia * vel +
                   self.c1 * r1 * (p_best_pos - pos) +
                   self.c2 * r2 * (g_best_pos - pos))
            pos = pos + vel
            pos = np.clip(pos, self.lower_bound, self.upper_bound)

            # Evaluate new positions
            new_vals = np.array([func(ind) for ind in pos])
            evaluations += self.pop_size

            # Update personal bests
            better_mask = new_vals < p_best_val
            p_best_pos[better_mask] = pos[better_mask]
            p_best_val[better_mask] = new_vals[better_mask]

            # Update global best
            if np.min(p_best_val) < g_best_val:
                g_best_pos = p_best_pos[np.argmin(p_best_val)]
                g_best_val = np.min(p_best_val)

            # Random Walk Mutation Strategy
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                walk_step = np.random.uniform(-0.1, 0.1, self.dim)
                trial = p_best_pos[i] + walk_step
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                trial_val = func(trial)
                evaluations += 1

                # Replace if trial is better
                if trial_val < p_best_val[i]:
                    p_best_pos[i] = trial
                    p_best_val[i] = trial_val
                    if trial_val < g_best_val:
                        g_best_pos = trial
                        g_best_val = trial_val

        return g_best_pos, g_best_val