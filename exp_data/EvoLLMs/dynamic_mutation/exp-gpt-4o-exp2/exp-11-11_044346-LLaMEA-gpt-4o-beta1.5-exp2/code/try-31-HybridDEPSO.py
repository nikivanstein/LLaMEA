import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = min(30, budget // 10)
        self.c1 = 2.0  # Increased cognitive component for individual influence
        self.c2 = 2.0  # Increased social component for collective influence
        self.w = 0.5   # Slightly adjusted initial inertia weight
        self.F = 0.6   # Reduced DE scaling factor for finer search
        self.CR = 0.9  # DE crossover probability

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
            # Dynamic inertia weight with random variation
            self.w = 0.5 + (0.9 - 0.5) * (1 - eval_count / self.budget) + np.random.uniform(-0.05, 0.05)

            # Evaluate the population
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                current_value = func(pop[i])
                eval_count += 1

                # Update personal best
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best[i] = pop[i].copy()

                # Update global best
                if current_value < best_global_value:
                    best_global_value = current_value
                    best_global_position = pop[i].copy()

            if eval_count >= self.budget:
                break

            # Perform dynamic DE mutation and crossover
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                indices = np.random.choice(self.population_size, 3, replace=False)
                r1, r2, r3 = indices
                while r1 == i or r2 == i or r3 == i:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    r1, r2, r3 = indices

                F_dynamic = self.F * (1 + np.random.uniform(-0.1, 0.1))  # Dynamic scaling factor
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

            # Update velocities and positions using PSO with enhanced influence
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - pop[i]) +
                                 self.c2 * r2 * (best_global_position - pop[i]))

                pop[i] = pop[i] + velocities[i]
                pop[i] = np.clip(pop[i], lower, upper)

        return best_global_position, best_global_value