import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.de_f = 0.8  # Differential weight
        self.de_cr = 0.9  # Crossover probability
        self.pso_w = 0.5  # Inertia weight
        self.pso_c1 = 0.8  # Cognitive coefficient
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
            # Differential Evolution mutation and crossover
            for i in range(self.population_size):
                if self.eval_count >= self.max_evaluations:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.de_f * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < self.de_cr
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
                velocity[i] = (self.pso_w * velocity[i] +
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

        return global_best, personal_best_fitness[global_best_idx]