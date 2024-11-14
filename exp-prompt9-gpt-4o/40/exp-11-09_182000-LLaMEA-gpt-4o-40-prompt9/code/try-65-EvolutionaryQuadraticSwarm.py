import numpy as np

class EvolutionaryQuadraticSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 35  # Smaller population for faster convergence
        self.w = 0.6  # Increased inertia for stable convergence
        self.cr = 0.85  # Slightly reduced crossover rate
        self.f1 = 0.4  # Reduced mutation factor for more controlled exploration
        self.q_prob = 0.15  # Enhanced quantum factor for diversity boost
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
                    quadratic_component = self.f1 * (x0 - x1) * (x2 - x3)
                    if np.random.rand() < 0.5:
                        mutant = x0 + quadratic_component
                    else:
                        mutant = x0 + self.f1 * (global_best - personal_best[i]) + self.f1 * (x1 - x2)

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

            self.w = 0.4 + 0.25 * (1 - evaluations / self.budget)  # Dynamic inertia adjustment
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 r1 * (personal_best[i] - population[i]) +
                                 r2 * (global_best - population[i]))
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