import numpy as np

class SelfAdaptiveQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50  # Increased population for better diversity
        self.w = 0.7  # Higher inertia for initial exploration
        self.cr = 0.85  # Adaptive crossover rate
        self.f1 = 0.4  # Lower mutation factor for finer search
        self.alpha = 0.5  # New gradient influence factor
        self.q_prob = 0.15  # Increased quantum probability for diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-0.1, 0.1, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]

        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, 4, replace=False)
                x0, x1, x2, x3 = population[indices]
                if np.random.rand() < self.q_prob:
                    quantum_shift = np.random.uniform(-1, 1, self.dim)
                    mutant = global_best + quantum_shift * (x1 - x2)
                else:
                    mutant = x0 + self.f1 * (x1 - x2) + self.f1 * (x2 - x3)

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

            self.w = 0.5 + 0.2 * (1 - evaluations / self.budget)  # Adaptive inertia adjustment
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                r1, r2 = np.random.rand(), np.random.rand()
                gradient = np.random.uniform(-0.1, 0.1, self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 r1 * (personal_best[i] - population[i]) +
                                 r2 * (global_best - population[i]) +
                                 self.alpha * gradient)
                population[i] += velocities[i]
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)

                f_val = func(population[i])
                evaluations += 1

                if f_val < personal_best_values[i]:
                    personal_best_values[i] = f_val
                    personal_best[i] = population[i].copy()

                    if f_val < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[global_best_idx]

        return global_best