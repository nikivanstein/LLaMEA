import numpy as np

class EnhancedMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.init_population_size = 10 * dim
        self.population_size = self.init_population_size
        self.mutation_factor = 0.9
        self.crossover_rate = 0.8
        self.adaptive_factor = 0.1
        self.learning_rate = 0.05
        self.history = []
        self.covariance_matrix = np.eye(dim)

    def _chaotic_initialization(self):
        population = np.zeros((self.population_size, self.dim))
        z = np.random.rand(self.dim)
        for i in range(self.population_size):
            z = 4 * z * (1 - z)
            population[i] = self.lb + z * (self.ub - self.lb)
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
            self.mutation_factor = np.clip(self.mutation_factor + adjustment, 0.6, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate + adjustment, 0.7, 1.0)

    def _resize_population(self, evaluations):
        if evaluations > self.budget * 0.2:
            self.population_size = max(5 * self.dim, self.init_population_size // 2)
        if evaluations > self.budget * 0.7:
            self.population_size = max(3 * self.dim, self.init_population_size // 4)

    def _update_covariance(self, population, centroid):
        deviations = population - centroid
        self.covariance_matrix = np.cov(deviations, rowvar=False) + np.eye(self.dim) * 1e-6

    def _adaptive_learning(self, current_best_fitness):
        if self.history:
            recent_improvement = (self.history[-1] - current_best_fitness) / abs(self.history[-1])
            if recent_improvement < 0.01:
                self.learning_rate = np.clip(self.learning_rate * 1.2, 0.01, 0.1)
            else:
                self.learning_rate = np.clip(self.learning_rate * 0.8, 0.01, 0.1)
        self.history.append(current_best_fitness)

    def __call__(self, func):
        population = self._chaotic_initialization()
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        evaluations = self.population_size
        centroid = np.mean(population, axis=0)

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
                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            self._update_covariance(population, centroid)
            levy_candidate = self._levy_flight(best_individual, centroid)
            levy_fitness = func(levy_candidate)
            evaluations += 1

            if levy_fitness < best_fitness:
                best_individual = levy_candidate
                best_fitness = levy_fitness

            self._resize_population(evaluations)
            self._adaptive_learning(best_fitness)

        return best_individual, best_fitness