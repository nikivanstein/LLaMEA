import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * int(np.sqrt(dim))  # population size
        self.inertia = 0.7  # inertia weight for PSO
        self.c1 = 1.4  # cognitive component
        self.c2 = 1.4  # social component
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        # Adaptive parameters
        self.mutation_factor = 0.8  # initial mutation factor
        self.crossover_rate = 0.9  # initial crossover rate
        self.chi = 0.5  # chaotic component

    def chaotic_map(self, x):
        return 4 * x * (1 - x)

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
            # PSO Update with chaotic dynamics
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            vel = (self.inertia * vel +
                   self.c1 * r1 * (p_best_pos - pos) +
                   self.c2 * r2 * (g_best_pos - pos) +
                   self.chi * np.random.uniform(-1, 1, self.dim))
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

            # Adaptive DE Mutation and Crossover
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = p_best_pos[idxs]
                m_factor = self.mutation_factor * (1 - evaluations / self.budget)
                c_rate = self.crossover_rate * (1 - evaluations / self.budget)
                mutant = a + m_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < c_rate
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