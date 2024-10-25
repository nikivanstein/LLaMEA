import numpy as np

class HarmonicAdaptiveEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, self.budget // 25)  # Adjusted population size
        self.inertia = np.random.uniform(0.4, 0.9)  # Adjusted inertia bounds
        self.c1, self.c2 = 1.5, 1.5  # Modified cognitive and social coefficients
        self.F = 0.6 + np.random.rand() * 0.2  # Modified mutation factor range
        self.CR = 0.7  # Adjusted crossover rate

    def __call__(self, func):
        lb, ub = -5.0, 5.0
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-0.6, 0.6, (self.pop_size, self.dim))
        personal_best = pop.copy()
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best_idx = np.argmin(personal_best_scores)
        global_best = personal_best[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        evaluations = self.pop_size

        while evaluations < self.budget:
            harmonic_factor = np.cos(np.pi * evaluations / self.budget)  # Modified harmonic factor
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.inertia * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - pop[i]) +
                                 self.c2 * r2 * (global_best - pop[i]) * harmonic_factor)
                pop[i] += velocities[i]
                np.clip(pop[i], lb, ub, out=pop[i])

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

            self.inertia = max(0.1, self.inertia * 0.95)  # Adjusted inertia decay
            self.F = 0.4 + np.random.rand() * 0.5  # Adjusted mutation factor range

        return global_best, global_best_score