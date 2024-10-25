import numpy as np

class AdaptiveMultiStrategyOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(100, self.budget // 10)  # Adjusted population size
        self.inertia = np.random.uniform(0.5, 0.9)  # Adaptive inertia weight
        self.c1, self.c2 = 1.5, 2.0  # Enhanced cognitive and social coefficients
        self.F = 0.8 + np.random.rand() * 0.4  # More dynamic scaling factor
        self.CR = 0.9  # Increased crossover probability

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
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.inertia * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - pop[i]) +
                                 self.c2 * r2 * (global_best - pop[i]))
                pop[i] += velocities[i]
                np.clip(pop[i], lb, ub, out=pop[i])

                # Hybrid Mutation Strategy: Differential Evolution and Gaussian
                if np.random.rand() < self.CR:
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.F * (b - c), lb, ub)
                    gaussian_noise = np.random.normal(0, 0.1, self.dim)
                    hybrid_mutant = np.clip(mutant + gaussian_noise, lb, ub)
                    trial = np.where(np.random.rand(self.dim) < self.CR, hybrid_mutant, pop[i])
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

            # Adaptive adjustment of inertia and F for exploration-exploitation balance
            self.inertia = max(0.4, self.inertia - 0.02 * (1 - evaluations / self.budget))
            self.F = 0.7 + np.random.rand() * 0.3

        return global_best, global_best_score