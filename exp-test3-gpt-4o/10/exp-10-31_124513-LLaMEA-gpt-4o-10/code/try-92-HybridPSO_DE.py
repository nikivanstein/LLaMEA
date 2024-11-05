import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.c1 = 2.0  # cognitive component
        self.c2 = 2.0  # social component
        self.w_max = 0.9  # max inertia weight
        self.w_min = 0.4  # min inertia weight
        self.F = 0.5  # differential evolution scaling factor
        self.CR = 0.9  # crossover probability
        self.lb = -5.0
        self.ub = 5.0
        self.vel_max = 0.5  # maximum velocity

    def __call__(self, func):
        # Adaptive swarm size
        initial_population_size = min(40, 5 * self.dim)  # Ensure swarm size scalability
        pos = np.random.uniform(self.lb, self.ub, (initial_population_size, self.dim))
        vel = np.random.uniform(-1, 1, (initial_population_size, self.dim))
        pbest = pos.copy()
        pbest_val = np.full(initial_population_size, np.inf)

        # Evaluate initial population
        eval_count = 0
        for i in range(initial_population_size):
            pbest_val[i] = func(pos[i])
            eval_count += 1

        # Determine the global best
        gbest_idx = np.argmin(pbest_val)
        gbest = pbest[gbest_idx]
        gbest_val = pbest_val[gbest_idx]

        chaos_val = 0.7  # Initial value for chaotic mapping
        while eval_count < self.budget:
            # Sinusoidal chaotic inertia weight
            chaos_val = 4 * chaos_val * (1 - chaos_val)  # Logistic map
            wave = 0.5 * np.sin(2 * np.pi * eval_count / self.budget)
            dynamic_factor = (self.budget - eval_count) / self.budget
            self.w = (self.w_max - (self.w_max - self.w_min) * (chaos_val + wave)) * dynamic_factor

            # Random perturbation in inertia weight
            self.w += np.random.uniform(-0.05, 0.05)

            for i in range(initial_population_size):
                # Adaptive cognitive and social components
                c1_adaptive = 1.5 + 0.5 * chaos_val * dynamic_factor  # Enhanced adaptability
                c2_adaptive = 1.5 + 0.5 * (1 - chaos_val) * dynamic_factor  # Enhanced adaptability
                r1, r2 = np.random.rand(2)
                vel[i] = (self.w * vel[i] 
                          + c1_adaptive * r1 * (pbest[i] - pos[i]) 
                          + c2_adaptive * r2 * (gbest - pos[i]))
                vel[i] = np.clip(vel[i], -self.vel_max, self.vel_max)  # Clamping velocity
                pos[i] = pos[i] + vel[i]

                # Enforce boundary constraints
                pos[i] = np.clip(pos[i], self.lb, self.ub)

                # Enhanced DE mutation diversity
                F_adaptive = 0.4 + 0.3 * chaos_val + np.random.uniform(-0.1, 0.1)

                # DE mutation and crossover with rotational permutation
                indices = [idx for idx in range(initial_population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = pos[a] + F_adaptive * (pos[b] - pos[(c+1)%len(indices)])
                mutant = np.clip(mutant, self.lb, self.ub)

                # Improved DE crossover diversity
                CR_adaptive = 0.8 + 0.2 * np.random.rand() * dynamic_factor
                trial = np.where(np.random.rand(self.dim) < CR_adaptive, mutant, pos[i])
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