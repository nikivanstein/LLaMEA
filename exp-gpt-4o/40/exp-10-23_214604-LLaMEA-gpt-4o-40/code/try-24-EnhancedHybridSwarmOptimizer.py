import numpy as np

class EnhancedHybridSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(100, self.budget // 10)
        self.inertia_min, self.inertia_max = 0.2, 0.9
        self.c1_start, self.c1_end = 2.5, 1.0
        self.c2_start, self.c2_end = 0.5, 2.0
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        lb, ub = -5.0, 5.0
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = pop.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best_idx = np.argmin(personal_best_scores)
        global_best = personal_best[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        evaluations = self.pop_size

        while evaluations < self.budget:
            w = self.inertia_max - (self.inertia_max - self.inertia_min) * (evaluations / self.budget)
            c1 = self.c1_start - (self.c1_start - self.c1_end) * (evaluations / self.budget)
            c2 = self.c2_start + (self.c2_end - self.c2_start) * (evaluations / self.budget)

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_best[i] - pop[i]) +
                                 c2 * r2 * (global_best - pop[i]))
                pop[i] += velocities[i]
                pop[i] = np.clip(pop[i], lb, ub)

                # Differential Evolution Mutation and Crossover
                if np.random.rand() < self.CR:
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.F * (b - c), lb, ub)
                    trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])
                    trial_score = func(trial)
                    evaluations += 1

                    if trial_score < personal_best_scores[i]:
                        personal_best[i] = trial
                        personal_best_scores[i] = trial_score

                        if trial_score < global_best_score:
                            global_best = trial
                            global_best_score = trial_score

                if evaluations >= self.budget:
                    break

        return global_best, global_best_score