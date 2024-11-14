import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = min(30, budget // 10)
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.9  # Adapted inertia weight for faster convergence
        self.F = 0.8
        self.CR = 0.9

    def __call__(self, func):
        np.random.seed(42)
        lower, upper = self.bounds
        pop = np.random.uniform(lower, upper, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_values = np.full(self.population_size, np.inf)

        eval_count = 0
        best_global_value = np.inf
        best_global_position = None

        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                current_value = func(pop[i])
                eval_count += 1

                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best[i] = pop[i].copy()

                if current_value < best_global_value:
                    best_global_value = current_value
                    best_global_position = pop[i].copy()

            if eval_count >= self.budget:
                break

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                indices = np.random.choice(self.population_size, 3, replace=False)
                r1, r2, r3 = indices
                while r1 == i or r2 == i or r3 == i:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    r1, r2, r3 = indices

                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lower, upper)

                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])

                trial_value = func(trial)
                eval_count += 1

                if trial_value < personal_best_values[i]:
                    personal_best_values[i] = trial_value
                    personal_best[i] = trial.copy()

                if trial_value < best_global_value:
                    best_global_value = trial_value
                    best_global_position = trial.copy()

            if eval_count >= self.budget:
                break

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - pop[i]) +
                                 self.c2 * r2 * (best_global_position - pop[i]))

                velocities[i] = np.clip(velocities[i], -2, 2)  # Dimension-wise velocity clamping

                pop[i] = pop[i] + velocities[i]
                pop[i] = np.clip(pop[i], lower, upper)

        return best_global_position, best_global_value