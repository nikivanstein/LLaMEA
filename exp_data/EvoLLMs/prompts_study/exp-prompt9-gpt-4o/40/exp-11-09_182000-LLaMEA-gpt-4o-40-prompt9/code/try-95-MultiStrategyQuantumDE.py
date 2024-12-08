import numpy as np

class MultiStrategyQuantumDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50  # Increased population size for better exploration
        self.f = 0.5  # Base mutation factor
        self.cr = 0.9  # Crossover rate
        self.q_prob = 0.15  # Increased quantum probability for diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]

        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, 5, replace=False)
                x0, x1, x2, x3, x4 = population[indices]
                if np.random.rand() < self.q_prob:
                    quantum_shift = np.random.uniform(-1, 1, self.dim)
                    mutant = global_best + quantum_shift * (x1 - x2)
                else:
                    adaptive_f = self.f + np.random.uniform(-0.1, 0.1)  # Adaptive scaling
                    if np.random.rand() < 0.5:
                        mutant = x0 + adaptive_f * (x1 - x2) + adaptive_f * (x2 - x3)
                    else:
                        mutant = x0 + adaptive_f * (x0 - personal_best[i]) + adaptive_f * (x1 - x4)

                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, population[i])
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                f_trial = func(trial)
                evaluations += 1

                if f_trial < personal_best_values[i]:
                    personal_best_values[i] = f_trial
                    personal_best[i] = trial.copy()

                    if f_trial < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[global_best_idx]

                if evaluations >= self.budget:
                    break

        return global_best