import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 25  # Increased population size for more diversity
        self.initial_de_f = 0.9  # Slightly increased Differential weight
        self.initial_de_cr = 0.8  # Slightly lowered Crossover probability for more exploration
        self.initial_pso_w = 0.4  # Lowered Inertia weight for quicker convergence
        self.pso_c1 = 1.1  # Increased Cognitive coefficient for better personal exploration
        self.pso_c2 = 1.3  # Increased Social coefficient for stronger attraction to global best
        self.lb = -5.0
        self.ub = 5.0
        self.max_evaluations = budget
        self.eval_count = 0

    def __call__(self, func):
        np.random.seed(42)
        self.eval_count = 0
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))  # Initialize with random velocities
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        global_best_idx = np.argmin(fitness)
        global_best = population[global_best_idx]

        while self.eval_count < self.max_evaluations:
            progress_ratio = self.eval_count / self.max_evaluations
            de_f = self.initial_de_f * (1 - progress_ratio) + 0.4 * progress_ratio
            de_cr = self.initial_de_cr * (1 - progress_ratio) + 0.5 * progress_ratio
            pso_w = self.initial_pso_w * (1 - progress_ratio) + 0.3 * progress_ratio

            for i in range(self.population_size):
                if self.eval_count >= self.max_evaluations:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + de_f * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < de_cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                self.eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < personal_best_fitness[i]:
                        personal_best[i] = trial
                        personal_best_fitness[i] = trial_fitness
                        if trial_fitness < personal_best_fitness[global_best_idx]:
                            global_best = trial
                            global_best_idx = i

            for i in range(self.population_size):
                if self.eval_count >= self.max_evaluations:
                    break
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocity[i] = (pso_w * velocity[i] +
                               self.pso_c1 * r1 * (personal_best[i] - population[i]) +
                               self.pso_c2 * r2 * (global_best - population[i]))
                population[i] = np.clip(population[i] + velocity[i], self.lb, self.ub)
                fitness[i] = func(population[i])
                self.eval_count += 1
                if fitness[i] < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = fitness[i]
                    if fitness[i] < personal_best_fitness[global_best_idx]:
                        global_best = population[i]
                        global_best_idx = i

            if self.eval_count >= self.max_evaluations * 0.70:
                diversity_count = max(1, int(self.population_size * 0.3))  # Increased diversity count
                randomized_indices = np.random.choice(self.population_size, diversity_count, replace=False)
                for idx in randomized_indices:
                    population[idx] = np.random.uniform(self.lb, self.ub, self.dim)
                    fitness[idx] = func(population[idx])
                    self.eval_count += 1
                    if fitness[idx] < personal_best_fitness[idx]:
                        personal_best[idx] = population[idx]
                        personal_best_fitness[idx] = fitness[idx]
                        if fitness[idx] < personal_best_fitness[global_best_idx]:
                            global_best = population[idx]
                            global_best_idx = idx

        return global_best, personal_best_fitness[global_best_idx]