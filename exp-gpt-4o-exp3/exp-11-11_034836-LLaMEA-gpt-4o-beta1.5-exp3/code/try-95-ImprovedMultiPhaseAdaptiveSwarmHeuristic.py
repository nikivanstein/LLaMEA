import numpy as np

class ImprovedMultiPhaseAdaptiveSwarmHeuristic:
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
        self.phase_switch_threshold = 0.25
        self.fitness_diversity_threshold = 1e-5
        self.history = []
        self.best_positions = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')

    def _initialize_population(self):
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        return population

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

    def _levy_flight(self, current_position, best_position):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / abs(v) ** (1 / beta)
        step_size = 0.01 * step * (current_position - best_position)
        return np.clip(current_position + step_size, self.lb, self.ub)

    def _adapt_parameters(self, fitness):
        median_fitness = np.median(fitness)
        fitness_variance = np.var(fitness)
        for idx, fit in enumerate(fitness):
            adjustment = self.adaptive_factor * (1 - 2 * (fit > median_fitness))
            self.mutation_factor = np.clip(self.mutation_factor + adjustment, 0.5, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate + adjustment, 0.7, 1.0)

        if fitness_variance < self.fitness_diversity_threshold:
            self.mutation_factor = np.clip(self.mutation_factor * 1.2, 0.5, 1.5)
            self.crossover_rate = np.clip(self.crossover_rate * 1.1, 0.7, 1.0)

    def _resize_population(self, evaluations):
        phase1_end = self.budget * self.phase_switch_threshold
        phase2_end = self.budget * (2 * self.phase_switch_threshold)
        if evaluations > phase1_end:
            self.population_size = max(6 * self.dim, self.init_population_size // 2)
        if evaluations > phase2_end:
            self.population_size = max(3 * self.dim, self.init_population_size // 4)

    def _adaptive_learning(self, current_best_fitness):
        if self.history:
            recent_improvement = (self.history[-1] - current_best_fitness) / (abs(self.history[-1]) + 1e-9)
            if recent_improvement < 0.01:
                self.learning_rate = np.clip(self.learning_rate * 1.05, 0.01, 0.1)
            else:
                self.learning_rate = np.clip(self.learning_rate * 0.95, 0.01, 0.1)
        self.history.append(current_best_fitness)

    def _update_global_best(self, candidate, candidate_fitness):
        if candidate_fitness < self.global_best_fitness:
            self.global_best_fitness = candidate_fitness
            self.global_best_position = candidate

    def _partition_population(self, population, fitness):
        sorted_indices = np.argsort(fitness)
        top_half = population[sorted_indices[:self.population_size // 2]]
        bottom_half = population[sorted_indices[self.population_size // 2:]]
        return top_half, bottom_half

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size

        while evaluations < self.budget:
            self._adapt_parameters(fitness)
            top_population, bottom_population = self._partition_population(population, fitness)

            for i in range(len(top_population)):
                if evaluations >= self.budget: break
                mutant = self._mutate(i, top_population)
                trial = self._crossover(top_population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    top_population[i] = trial
                    fitness[i] = trial_fitness
                    self._update_global_best(trial, trial_fitness)

            for i in range(len(bottom_population)):
                if evaluations >= self.budget: break
                mutant = self._mutate(i, bottom_population)
                trial = self._crossover(bottom_population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[self.population_size // 2 + i]:
                    bottom_population[i] = trial
                    fitness[self.population_size // 2 + i] = trial_fitness
                    self._update_global_best(trial, trial_fitness)

            population = np.vstack((top_population, bottom_population))
            centroid = np.mean(population, axis=0)
            levy_candidate = self._levy_flight(self.global_best_position, centroid)
            levy_fitness = func(levy_candidate)
            evaluations += 1

            if levy_fitness < self.global_best_fitness:
                self._update_global_best(levy_candidate, levy_fitness)

            self._resize_population(evaluations)
            self._adaptive_learning(self.global_best_fitness)

        return self.global_best_position, self.global_best_fitness