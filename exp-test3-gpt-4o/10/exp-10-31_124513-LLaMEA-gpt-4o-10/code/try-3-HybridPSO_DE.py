import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.c1 = 2.0  # cognitive component
        self.c2 = 2.0  # social component
        self.w = 0.7  # inertia weight
        self.F = 0.5  # differential evolution scaling factor
        self.CR = 0.9  # crossover probability
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        # Initialize particles
        pos = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        vel = np.random.uniform(-1, 1, (self.population_size, self.dim))
        pbest = pos.copy()
        pbest_val = np.full(self.population_size, np.inf)

        # Evaluate initial population
        eval_count = 0
        for i in range(self.population_size):
            pbest_val[i] = func(pos[i])
            eval_count += 1

        # Determine the global best
        gbest_idx = np.argmin(pbest_val)
        gbest = pbest[gbest_idx]
        gbest_val = pbest_val[gbest_idx]

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Update velocities and positions
                r1, r2 = np.random.rand(2)
                vel[i] = (self.w * vel[i] 
                          + self.c1 * r1 * (pbest[i] - pos[i]) 
                          + self.c2 * r2 * (gbest - pos[i]))
                pos[i] = pos[i] + vel[i]

                # Enforce boundary constraints
                pos[i] = np.clip(pos[i], self.lb, self.ub)

                # DE mutation and crossover
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = pos[a] + self.F * (pos[b] - pos[c])
                mutant = np.clip(mutant, self.lb, self.ub)

                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pos[i])
                trial_value = func(trial)
                eval_count += 1

                # Selection between trial and original
                if trial_value < pbest_val[i]:
                    pbest[i] = trial
                    pbest_val[i] = trial_value
                    if trial_value < gbest_val:
                        gbest = trial
                        gbest_val = trial_value

                # Break if the budget is exhausted
                if eval_count >= self.budget:
                    break

        return gbest, gbest_val