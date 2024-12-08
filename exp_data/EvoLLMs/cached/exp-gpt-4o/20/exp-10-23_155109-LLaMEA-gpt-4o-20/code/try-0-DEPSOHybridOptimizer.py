import numpy as np

class DEPSOHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size
        self.current_eval = 0
        self.bounds = (-5.0, 5.0)
        self.w = 0.5  # Inertia weight for PSO
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.F = 0.8  # Differential weight for DE
        self.CR = 0.9  # Crossover probability for DE

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = pop.copy()
        personal_best_values = np.array([func(ind) for ind in pop])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)

        while self.current_eval < self.budget:
            # Perform Differential Evolution steps
            for i in range(self.pop_size):
                if self.current_eval >= self.budget:
                    break
                # Mutation
                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                x0, x1, x2 = pop[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_value = func(trial)
                self.current_eval += 1
                if trial_value < personal_best_values[i]:
                    personal_best[i] = trial
                    personal_best_values[i] = trial_value
                    if trial_value < global_best_value:
                        global_best = trial
                        global_best_value = trial_value

            # Perform Particle Swarm Optimization steps
            for i in range(self.pop_size):
                if self.current_eval >= self.budget:
                    break
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - pop[i]) +
                                 self.c2 * r2 * (global_best - pop[i]))
                velocities[i] = np.clip(velocities[i], self.bounds[0] - pop[i], self.bounds[1] - pop[i])

                # Update position
                pop[i] = np.clip(pop[i] + velocities[i], self.bounds[0], self.bounds[1])
                value = func(pop[i])
                self.current_eval += 1

                # Update personal and global bests
                if value < personal_best_values[i]:
                    personal_best[i] = pop[i]
                    personal_best_values[i] = value
                    if value < global_best_value:
                        global_best = pop[i]
                        global_best_value = value

        return global_best