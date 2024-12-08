import numpy as np

class AdaptiveMultiSwarmPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20
        self.swarm_count = 3  # Multiple swarms
        self.w = 0.9
        self.c1 = 2.05
        self.c2 = 2.05
        self.F = 0.6
        self.CR = 0.8
        self.exploration_prob = 0.2
        self.adaptive_pop_size = self.pop_size

    def __call__(self, func):
        np.random.seed(42)
        pos = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        vel = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = pos.copy()
        personal_best_value = np.array([func(ind) for ind in personal_best])
        global_best_index = np.argmin(personal_best_value)
        global_best = personal_best[global_best_index]
        global_best_value = personal_best_value[global_best_index]

        evaluations = self.pop_size

        while evaluations < self.budget:
            self.w = 0.9 - 0.4 * (evaluations / self.budget)

            # Strategic exploration
            if np.random.rand() < self.exploration_prob:
                for i in range(self.swarm_count):
                    if evaluations >= self.budget:
                        break
                    idxs = np.random.choice(self.pop_size, self.pop_size // self.swarm_count, replace=False)
                    swarm_best_idx = idxs[np.argmin(personal_best_value[idxs])]
                    local_best = personal_best[swarm_best_idx]
                    r1 = np.random.rand(len(idxs), self.dim)
                    r2 = np.random.rand(len(idxs), self.dim)
                    vel[idxs] = self.w * vel[idxs] + self.c1 * r1 * (personal_best[idxs] - pos[idxs]) + self.c2 * r2 * (local_best - pos[idxs])
                    pos[idxs] = np.clip(pos[idxs] + vel[idxs], self.lower_bound, self.upper_bound)

            fitness = np.array([func(ind) for ind in pos])
            evaluations += self.pop_size

            better_mask = fitness < personal_best_value
            personal_best[better_mask] = pos[better_mask]
            personal_best_value[better_mask] = fitness[better_mask]
            
            current_global_best_index = np.argmin(personal_best_value)
            current_global_best_value = personal_best_value[current_global_best_index]
            if current_global_best_value < global_best_value:
                global_best = personal_best[current_global_best_index]
                global_best_value = current_global_best_value

            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pos[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pos[i])

                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    pos[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < global_best_value:
                        global_best = trial
                        global_best_value = trial_fitness

        return global_best, global_best_value