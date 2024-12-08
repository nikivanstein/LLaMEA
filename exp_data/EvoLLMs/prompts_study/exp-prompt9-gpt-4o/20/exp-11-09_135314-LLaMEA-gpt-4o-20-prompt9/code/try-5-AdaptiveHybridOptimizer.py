import numpy as np

class AdaptiveHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(40, budget // 10)
        self.strategy_switch = 0.3  # Switch strategy later in the budget

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        velocities = np.random.uniform(
            -0.5, 0.5, (self.population_size, self.dim)
        )
        fitness = np.apply_along_axis(func, 1, population)
        evals = self.population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()

        while evals < self.budget:
            if evals < self.strategy_switch * self.budget:
                # Particle Swarm Optimization approach
                inertia = 0.7
                cognitive_coeff = 1.5
                social_coeff = 1.5
                for i in range(self.population_size):
                    r1, r2 = np.random.rand(), np.random.rand()
                    velocities[i] = (
                        inertia * velocities[i]
                        + cognitive_coeff * r1 * (personal_best[i] - population[i])
                        + social_coeff * r2 * (best_solution - population[i])
                    )
                    population[i] += velocities[i]
                    population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
                    candidate_fitness = func(population[i])
                    evals += 1
                    if candidate_fitness < personal_best_fitness[i]:
                        personal_best[i] = population[i].copy()
                        personal_best_fitness[i] = candidate_fitness
                    if candidate_fitness < best_fitness:
                        best_fitness = candidate_fitness
                        best_solution = population[i].copy()
                    if evals >= self.budget:
                        break
            else:
                # Differential Evolution with adaptive scaling factor
                scale_factor = 0.9 - 0.4 * (evals / self.budget)
                for i in range(self.population_size):
                    a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                    mutant = a + scale_factor * (b - c)
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    trial = np.where(np.random.rand(self.dim) < 0.9, mutant, population[i])
                    trial_fitness = func(trial)
                    evals += 1
                    if trial_fitness < fitness[i]:
                        population[i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < best_fitness:
                            best_fitness = trial_fitness
                            best_solution = trial.copy()
                    if evals >= self.budget:
                        break

        return best_solution