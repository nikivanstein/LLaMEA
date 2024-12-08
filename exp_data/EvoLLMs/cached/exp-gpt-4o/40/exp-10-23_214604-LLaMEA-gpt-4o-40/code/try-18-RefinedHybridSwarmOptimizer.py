import numpy as np

class RefinedHybridSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(100, self.budget // 10)
        self.inertia = 0.7  # Adjusted for better exploration
        self.c1, self.c2 = 1.4, 1.7  # Fine-tuned learning factors
        self.F = 0.9  # Increased differential weight
        self.CR = 0.95  # Higher crossover rate to incorporate more diverse mutations

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
            adaptive_inertia = 0.9 - (0.5 * evaluations / self.budget)  # Adaptive inertia factor
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (adaptive_inertia * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - pop[i]) +
                                 self.c2 * r2 * (global_best - pop[i]))
                pop[i] += velocities[i]
                np.clip(pop[i], lb, ub, out=pop[i])

                # Differential Evolution Mutation and Crossover with adaptive F
                if np.random.rand() < self.CR:
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    adaptive_F = 0.8 + 0.2 * (evaluations / self.budget)  # Adaptive differential weight
                    mutant = np.clip(a + adaptive_F * (b - c), lb, ub)
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