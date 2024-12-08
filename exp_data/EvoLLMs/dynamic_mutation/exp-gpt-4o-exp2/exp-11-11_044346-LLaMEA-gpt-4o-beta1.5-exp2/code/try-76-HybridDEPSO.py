import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = min(30, budget // 10)
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.w = 0.7   # Initial inertia weight
        self.F = 0.8   # DE scaling factor
        self.CR_initial = 0.9  # Initial DE crossover probability

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        lower, upper = self.bounds
        pop = np.random.uniform(lower, upper, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = pop.copy()
        personal_best_values = np.full(self.population_size, np.inf)

        eval_count = 0
        best_global_value = np.inf
        best_global_position = None

        while eval_count < self.budget:
            self.w = 0.4 + (0.7 - 0.4) * (1 - eval_count / self.budget)
            if best_global_value < np.inf:
                self.w *= 0.9

            self.CR = self.CR_initial - 0.4 * eval_count / self.budget

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

                # Adjusting DE scaling factor based on evaluations
                F_dynamic = self.F - 0.2 * eval_count / self.budget
                mutant = pop[r1] + F_dynamic * (pop[r2] - pop[r3])
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
                
                # Dynamic social component
                c2_dynamic = self.c2 * (1 + eval_count / self.budget)

                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - pop[i]) +
                                 c2_dynamic * r2 * (best_global_position - pop[i]))

                pop[i] = pop[i] + velocities[i]
                pop[i] = np.clip(pop[i], lower, upper)

        return best_global_position, best_global_value