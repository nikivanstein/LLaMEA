import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * int(np.sqrt(dim))  # population size
        self.inertia = 0.9  # start with higher inertia weight
        self.c1 = 1.4  # cognitive component
        self.c2 = 1.4  # social component
        self.mutation_factor = 0.8  # mutation factor for DE
        self.crossover_rate = 0.9  # crossover rate for DE
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
        success_factor = 0.1  # initial success factor

        while evaluations < self.budget:
            # Adaptive inertia weight
            self.inertia = max(0.4, self.inertia * (1 - success_factor))

            # PSO Update
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
            success_factor = np.mean(better_mask)  # success measure

            # Update global best
            if np.min(p_best_val) < g_best_val:
                g_best_pos = p_best_pos[np.argmin(p_best_val)]
                g_best_val = np.min(p_best_val)

            # Self-adaptive DE Mutation and Crossover
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = p_best_pos[idxs]
                mutation_factor = np.random.rand() * 0.6 + 0.5  # self-adaptive mutation factor
                mutant = a + mutation_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover_mask, mutant, pos[i])
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