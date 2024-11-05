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
    
    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / abs(v) ** (1 / beta)
        return 0.01 * step * (L - self.lb)

    def __call__(self, func):
        pos = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        vel = np.random.uniform(-1, 1, (self.population_size, self.dim))
        pbest = pos.copy()
        pbest_val = np.full(self.population_size, np.inf)

        eval_count = 0
        for i in range(self.population_size):
            pbest_val[i] = func(pos[i])
            eval_count += 1

        gbest_idx = np.argmin(pbest_val)
        gbest = pbest[gbest_idx]
        gbest_val = pbest_val[gbest_idx]

        chaos_val = 0.7
        while eval_count < self.budget:
            chaos_val = 4 * chaos_val * (1 - chaos_val)
            wave = 0.5 * np.sin(2 * np.pi * eval_count / self.budget)
            self.w = self.w_max - (self.w_max - self.w_min) * (chaos_val + wave)

            for i in range(self.population_size):
                c1_adaptive = 1.5 + 0.5 * chaos_val
                c2_adaptive = 1.5 + 0.5 * (1 - chaos_val)
                r1, r2 = np.random.rand(2)
                vel[i] = (self.w * vel[i] 
                          + c1_adaptive * r1 * (pbest[i] - pos[i]) 
                          + c2_adaptive * r2 * (gbest - pos[i]))
                vel[i] = np.clip(vel[i], -self.vel_max, self.vel_max)
                pos[i] = pos[i] + vel[i]

                if np.random.rand() < 0.1:
                    pos[i] += self.levy_flight(pos[i])  # Applying Lévy flight

                pos[i] = np.clip(pos[i], self.lb, self.ub)

                F_adaptive = 0.4 + 0.3 * chaos_val

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = pos[a] + F_adaptive * (pos[b] - pos[c])
                mutant = np.clip(mutant, self.lb, self.ub)

                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pos[i])
                trial_value = func(trial)
                eval_count += 1

                if trial_value < pbest_val[i]:
                    pbest[i] = trial
                    pbest_val[i] = trial_value
                    if trial_value < gbest_val:
                        gbest = trial
                        gbest_val = trial_value

                if eval_count >= self.budget:
                    break

        return gbest, gbest_val