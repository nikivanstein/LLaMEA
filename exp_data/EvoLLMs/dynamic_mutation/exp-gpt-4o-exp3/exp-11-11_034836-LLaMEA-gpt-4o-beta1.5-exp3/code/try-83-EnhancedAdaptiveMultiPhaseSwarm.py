import numpy as np

class EnhancedAdaptiveMultiPhaseSwarm:
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
        self.history = []
        self.best_positions = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.memory = []

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
        for idx, fit in enumerate(fitness):
            adjustment = self.adaptive_factor * (1 - 2 * (fit > median_fitness))
            self.mutation_factor = np.clip(self.mutation_factor + adjustment, 0.5, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate + adjustment, 0.7, 1.0)

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
            self.memory.append((candidate, candidate_fitness))
            if len(self.memory) > 5:  # Retain only the last 5 best solutions
                self.memory.pop(0)

    def _progressive_memory(self, population, fitness):
        for pos, fit in self.memory:
            if fit < self.global_best_fitness:
                population[np.argmax(fitness)] = pos
                fitness[np.argmax(fitness)] = fit

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size

        while evaluations < self.budget:
            self._adapt_parameters(fitness)
            for i in range(self.population_size):
                mutant = self._mutate(i, population)
                trial = self._crossover(population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    self._update_global_best(trial, trial_fitness)

                if evaluations >= self.budget:
                    break

            centroid = np.mean(population, axis=0)
            levy_candidate = self._levy_flight(self.global_best_position, centroid)
            levy_fitness = func(levy_candidate)
            evaluations += 1

            if levy_fitness < self.global_best_fitness:
                self._update_global_best(levy_candidate, levy_fitness)

            self._resize_population(evaluations)
            self._adaptive_learning(self.global_best_fitness)
            self._progressive_memory(population, fitness)

        return self.global_best_position, self.global_best_fitness