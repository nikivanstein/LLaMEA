import numpy as np

class EnhancedAdaptiveDEPSOHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 12 * dim  # Adjusted population size for better exploration
        self.current_eval = 0
        self.bounds = (-5.0, 5.0)
        self.w = 0.65  # Further adjusted inertia weight for control
        self.c1 = 1.7  # Altered cognitive coefficient for enhanced local search
        self.c2 = 1.6  # Balanced social coefficient
        self.F = 0.85  # Fine-tuned mutation factor for diversity
        self.CR = 0.8  # Adjusted crossover rate for explorative robustness
        self.adapt_factor = 0.9  # Adaptation factor for dynamic control
        self.diversity_prob = 0.25  # Increased probability for diversity manipulation

    def __call__(self, func):
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = pop.copy()
        personal_best_values = np.array([func(ind) for ind in pop])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        while self.current_eval < self.budget:
            self.w *= self.adapt_factor  # Continuously adjust inertia weight

            for i in range(self.pop_size):
                if self.current_eval >= self.budget:
                    break

                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                x0, x1, x2 = pop[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, pop[i])
                trial_value = func(trial)
                self.current_eval += 1

                if trial_value < personal_best_values[i]:
                    personal_best[i] = trial
                    personal_best_values[i] = trial_value
                    if trial_value < global_best_value:
                        global_best = trial
                        global_best_value = trial_value

            for i in range(self.pop_size):
                if self.current_eval >= self.budget:
                    break
                
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - pop[i]) +
                                 self.c2 * r2 * (global_best - pop[i]))
                velocities[i] = np.clip(velocities[i], self.bounds[0] - pop[i], self.bounds[1] - pop[i])

                pop[i] = np.clip(pop[i] + velocities[i], self.bounds[0], self.bounds[1])
                value = func(pop[i])
                self.current_eval += 1

                if value < personal_best_values[i]:
                    personal_best[i] = pop[i]
                    personal_best_values[i] = value
                    if value < global_best_value:
                        global_best = pop[i]
                        global_best_value = value

            if self.current_eval >= self.budget:
                break

            if np.random.rand() < self.diversity_prob:
                for j in range(self.pop_size):
                    challenger = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
                    challenger_value = func(challenger)
                    self.current_eval += 1
                    if challenger_value < personal_best_values[j]:
                        personal_best[j] = challenger
                        personal_best_values[j] = challenger_value
                        if challenger_value < global_best_value:
                            global_best = challenger
                            global_best_value = challenger_value

        return global_best