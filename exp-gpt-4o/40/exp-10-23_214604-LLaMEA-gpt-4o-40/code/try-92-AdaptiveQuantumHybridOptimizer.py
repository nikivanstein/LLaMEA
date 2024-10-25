import numpy as np

class AdaptiveQuantumHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(80, self.budget // 20)  # Slightly increased population size
        self.inertia = np.random.uniform(0.6, 1.2)  # Adjusted inertia range for exploration
        self.c1, self.c2 = 1.5, 1.5  # Balanced convergence and exploration
        self.F = 0.6 + np.random.rand() * 0.4  # Focused differential weight
        self.CR = 0.85  # Slightly reduced crossover rate for diversity

    def __call__(self, func):
        lb, ub = -5.0, 5.0
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-0.7, 0.7, (self.pop_size, self.dim))  # Increased initial velocity range
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

                if np.random.rand() < self.CR:
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    quantum_factor = np.random.uniform(-1, 1, self.dim)  # Quantum-inspired factor
                    mutant = np.clip(a + self.F * (b - c) + quantum_factor, lb, ub)
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

            self.inertia = max(0.3, self.inertia * 0.92)  # Adjusted inertia decay
            self.F = 0.5 + np.random.rand() * 0.5  # Dynamic adjustment of F

        return global_best, global_best_score