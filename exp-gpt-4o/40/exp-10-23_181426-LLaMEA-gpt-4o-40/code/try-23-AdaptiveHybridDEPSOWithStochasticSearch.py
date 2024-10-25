import numpy as np

class AdaptiveHybridDEPSOWithStochasticSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.initial_de_f = 0.8  # Differential weight
        self.initial_de_cr = 0.85  # Crossover probability
        self.initial_pso_w = 0.5  # Inertia weight
        self.pso_c1 = 1.0  # Cognitive coefficient
        self.pso_c2 = 1.2  # Social coefficient
        self.lb = -5.0
        self.ub = 5.0
        self.max_evaluations = budget
        self.eval_count = 0

    def __call__(self, func):
        np.random.seed(42)
        self.eval_count = 0
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += self.population_size
        personal_best = population.copy()
        personal_best_fitness = fitness.copy()
        global_best_idx = np.argmin(fitness)
        global_best = population[global_best_idx]

        while self.eval_count < self.max_evaluations:
            # Adjust DE and PSO parameters based on progress
            progress_ratio = self.eval_count / self.max_evaluations
            de_f = self.initial_de_f * (1 - progress_ratio) + 0.6 * progress_ratio
            de_cr = self.initial_de_cr * (1 - progress_ratio) + 0.7 * progress_ratio
            pso_w = self.initial_pso_w * (1 - progress_ratio) + 0.3 * progress_ratio

            # Differential Evolution mutation and crossover
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

            # Particle Swarm Optimization velocity and position update
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

            # Stochastic local search to enhance exploitation
            if self.eval_count >= self.max_evaluations * 0.80:
                for i in range(self.population_size):
                    if self.eval_count >= self.max_evaluations:
                        break
                    if np.random.rand() < 0.3:  # Randomly select individuals for local search
                        perturbation = np.random.normal(0, 0.1, self.dim)
                        candidate = np.clip(population[i] + perturbation, self.lb, self.ub)
                        candidate_fitness = func(candidate)
                        self.eval_count += 1
                        if candidate_fitness < fitness[i]:
                            population[i] = candidate
                            fitness[i] = candidate_fitness
                            if candidate_fitness < personal_best_fitness[i]:
                                personal_best[i] = candidate
                                personal_best_fitness[i] = candidate_fitness
                                if candidate_fitness < personal_best_fitness[global_best_idx]:
                                    global_best = candidate
                                    global_best_idx = i

        return global_best, personal_best_fitness[global_best_idx]