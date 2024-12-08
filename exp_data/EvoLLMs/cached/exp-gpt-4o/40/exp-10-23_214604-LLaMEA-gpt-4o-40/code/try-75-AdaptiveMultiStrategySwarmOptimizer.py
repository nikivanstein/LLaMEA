import numpy as np

class AdaptiveMultiStrategySwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(70, self.budget // 10)  # Adjusted population size with slight reduction
        self.base_inertia = np.random.uniform(0.3, 0.8)  # Lowered base adaptive inertia
        self.c1, self.c2 = 1.4, 1.6  # Modified cognitive and social coefficients for more balance
        self.F_base = 0.5  # Base scaling factor for mutation
        self.CR = 0.9  # Higher crossover probability for more aggressive mixing

    def __call__(self, func):
        lb, ub = -5.0, 5.0
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.pop_size, self.dim))  # Adjusted velocity range
        personal_best = pop.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best_idx = np.argmin(personal_best_scores)
        global_best = personal_best[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                inertia_adaptive = self.base_inertia + 0.5 * (global_best_score - personal_best_scores[i]) / (1 + abs(global_best_score))
                velocities[i] = (inertia_adaptive * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - pop[i]) +
                                 self.c2 * r2 * (global_best - pop[i]))
                pop[i] += velocities[i]
                np.clip(pop[i], lb, ub, out=pop[i])

                if np.random.rand() < self.CR:
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.F_base * (b - c) * np.random.rand(), lb, ub)
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

            if evaluations % (self.pop_size * 2) == 0:  # Periodic adaptation
                self.base_inertia = max(0.3, self.base_inertia - 0.02)
                self.F_base = 0.5 + np.random.rand() * 0.4

        return global_best, global_best_score