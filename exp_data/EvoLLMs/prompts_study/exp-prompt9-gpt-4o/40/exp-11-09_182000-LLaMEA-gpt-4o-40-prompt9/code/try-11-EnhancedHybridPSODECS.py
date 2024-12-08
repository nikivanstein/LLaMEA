import numpy as np

class EnhancedHybridPSODECS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 60  # Increased population size for better diversity
        self.c1 = 1.5  # Adjusted cognitive parameter for balanced convergence
        self.c2 = 1.5  # Adjusted social parameter for balanced convergence
        self.w = 0.6   # Slightly increased inertia for better exploration
        self.pa = 0.3  # Higher discovery rate for Cuckoo Search
        self.f = 0.8   # Differential evolution mutation factor
        self.cr = 0.9  # Crossover probability for differential evolution
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize swarm
        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best = swarm.copy()
        personal_best_values = np.array([func(ind) for ind in swarm])
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]

        evaluations = self.pop_size

        while evaluations < self.budget:
            # PSO Update
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - swarm[i]) +
                                 self.c2 * r2 * (global_best - swarm[i]))
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                f_val = func(swarm[i])
                evaluations += 1

                # Update personal and global bests
                if f_val < personal_best_values[i]:
                    personal_best_values[i] = f_val
                    personal_best[i] = swarm[i].copy()

                    if f_val < personal_best_values[global_best_idx]:
                        global_best_idx = i
                        global_best = personal_best[global_best_idx]

                if evaluations >= self.budget:
                    break

            # Differential Evolution inspired mutation and crossover
            for i in range(self.pop_size):
                if np.random.rand() < self.cr:
                    idxs = np.random.choice(self.pop_size, 3, replace=False)
                    x0, x1, x2 = swarm[idxs[0]], swarm[idxs[1]], swarm[idxs[2]]
                    mutant_vector = x0 + self.f * (x1 - x2)
                    mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                    
                    if func(mutant_vector) < personal_best_values[i]:
                        personal_best[i] = mutant_vector.copy()
                        personal_best_values[i] = func(mutant_vector)
                        evaluations += 1

                        if personal_best_values[i] < personal_best_values[global_best_idx]:
                            global_best_idx = i
                            global_best = personal_best[global_best_idx]

                if evaluations >= self.budget:
                    break

            # Cuckoo Search (CS) inspired update
            for i in range(self.pop_size):
                if np.random.rand() < self.pa:
                    step_size = np.random.standard_cauchy(size=self.dim)
                    new_solution = swarm[i] + step_size * (swarm[i] - global_best)
                    new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)

                    f_new = func(new_solution)
                    evaluations += 1

                    if f_new < personal_best_values[i]:
                        personal_best_values[i] = f_new
                        personal_best[i] = new_solution.copy()

                        if f_new < personal_best_values[global_best_idx]:
                            global_best_idx = i
                            global_best = personal_best[global_best_idx]

                if evaluations >= self.budget:
                    break

        return global_best