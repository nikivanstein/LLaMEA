import numpy as np

class AdaptiveHybridOptimization:
    def __init__(self, budget, dim, pop_size=50, F_base=0.7, CR_base=0.85, inertia_weight=0.7):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_base = F_base
        self.CR_base = CR_base
        self.inertia_weight = inertia_weight
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = population.copy()
        personal_best_fitness = fitness.copy()
        global_best_idx = np.argmin(fitness)
        global_best = population[global_best_idx].copy()

        evaluations = self.pop_size
        while evaluations < self.budget:
            for i in range(self.pop_size):
                r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
                inertia_dynamic = self.inertia_weight * (0.9 + 0.1 * (1 - (evaluations / self.budget)))
                velocities[i] = (inertia_dynamic * velocities[i] +
                                 r1 * (personal_best_positions[i] - population[i]) +
                                 r2 * (global_best - population[i]))
                population[i] = np.clip(population[i] + velocities[i], self.lower_bound, self.upper_bound)

                F_dynamic = self.F_base + 0.2 * (np.random.rand() - 0.5) * (personal_best_fitness[i] / (fitness[i] + 1e-8))
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + F_dynamic * (x1 - x2), self.lower_bound, self.upper_bound)

                CR_dynamic = self.CR_base + 0.3 * (np.random.rand() - 0.5)
                cross_points = np.random.rand(self.dim) < CR_dynamic
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best_positions[i] = trial
                        personal_best_fitness[i] = trial_fitness
                        if trial_fitness < fitness[global_best_idx]:
                            global_best_idx = i
                            global_best = trial.copy()

                if evaluations >= self.budget:
                    break

                if np.random.rand() < 0.5 and evaluations % (self.pop_size // 3) == 0:
                    local_search_idx = np.random.choice(self.pop_size)
                    local_solution = population[local_search_idx]
                    perturbation = np.random.normal(0, 0.15, self.dim) + 0.1 * velocities[local_search_idx]
                    local_mutant = np.clip(local_solution + perturbation, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_mutant)
                    evaluations += 1
                    if local_fitness < fitness[local_search_idx]:
                        population[local_search_idx] = local_mutant
                        fitness[local_search_idx] = local_fitness
                        if local_fitness < fitness[global_best_idx]:
                            global_best_idx = local_search_idx
                            global_best = local_mutant.copy()

        return global_best