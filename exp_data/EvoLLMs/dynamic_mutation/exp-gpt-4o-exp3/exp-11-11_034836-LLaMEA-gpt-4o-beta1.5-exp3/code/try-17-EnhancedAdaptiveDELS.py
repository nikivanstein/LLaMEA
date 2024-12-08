import numpy as np

class EnhancedAdaptiveDELS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.init_population_size = 10 * dim
        self.population_size = self.init_population_size
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.adaptive_factor = 0.1
        self.learning_rate = 0.05
        self.convergence_threshold = 1e-6

    def _initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

    def _mutate(self, target_idx, population):
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        mutant = a + self.mutation_factor * (b - c)
        return np.clip(mutant, self.lb, self.ub)

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _hybrid_local_global_search(self, individual, best_individual):
        global_step = np.random.normal(0, self.learning_rate, self.dim)
        local_step = np.random.uniform(-0.1, 0.1, self.dim)
        candidate = individual + global_step + local_step * (best_individual - individual)
        return np.clip(candidate, self.lb, self.ub)

    def _adapt_parameters(self, fitness, previous_best_fitness):
        median_fitness = np.median(fitness)
        for idx, fit in enumerate(fitness):
            adjustment = self.adaptive_factor * (1 - 2 * (fit > median_fitness))
            self.mutation_factor = np.clip(self.mutation_factor + adjustment, 0.5, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate + adjustment, 0.7, 1.0)
            self.learning_rate = np.clip(self.learning_rate + 0.01 * adjustment, 0.01, 0.1)

    def _dynamic_strategy(self, current_best_fitness, previous_best_fitness):
        if np.abs(current_best_fitness - previous_best_fitness) < self.convergence_threshold:
            self.mutation_factor = 1.0
            self.crossover_rate = 1.0
        else:
            self.mutation_factor = 0.8
            self.crossover_rate = 0.9

    def _resize_population(self, evaluations):
        if evaluations > self.budget * 0.5:
            self.population_size = max(4 * self.dim, self.init_population_size // 4)

    def _update_covariance(self, population, centroid):
        deviations = population - centroid
        self.covariance_matrix = np.cov(deviations, rowvar=False)
        self.covariance_matrix += np.eye(self.dim) * 1e-6  # Regularization

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        previous_best_fitness = best_fitness
        evaluations = self.population_size
        centroid = np.mean(population, axis=0)
        self.covariance_matrix = np.eye(self.dim)

        while evaluations < self.budget:
            self._adapt_parameters(fitness, previous_best_fitness)
            self._dynamic_strategy(best_fitness, previous_best_fitness)
            previous_best_fitness = best_fitness

            for i in range(self.population_size):
                mutant = self._mutate(i, population)
                trial = self._crossover(population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            self._update_covariance(population, centroid)
            hybrid_candidate = self._hybrid_local_global_search(best_individual, np.mean(population, axis=0))
            hybrid_fitness = func(hybrid_candidate)
            evaluations += 1

            if hybrid_fitness < best_fitness:
                best_individual = hybrid_candidate
                best_fitness = hybrid_fitness

            self._resize_population(evaluations)

        return best_individual, best_fitness