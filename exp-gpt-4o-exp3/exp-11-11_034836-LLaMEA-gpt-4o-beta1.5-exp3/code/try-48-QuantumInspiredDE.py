import numpy as np

class QuantumInspiredDE:
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
        self.covariance_matrix = np.eye(dim)
        self.history = []

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

    def _quantum_superposition(self, population, best_individual):
        phase_shift = self.learning_rate * np.random.randn(self.dim)
        quantum_states = np.exp(1j * phase_shift)
        quantum_population = population + np.real(quantum_states) * (best_individual - population)
        return np.clip(quantum_population.real, self.lb, self.ub)

    def _orthogonal_exploration(self, population):
        orthogonal_population = np.zeros_like(population)
        for i in range(self.dim):
            permutation = np.random.permutation(self.dim)
            orthogonal_population[:, i] = population[:, permutation[i]]
        return orthogonal_population

    def _adapt_parameters(self, fitness):
        median_fitness = np.median(fitness)
        for idx, fit in enumerate(fitness):
            adjustment = self.adaptive_factor * (1 - 2 * (fit > median_fitness))
            self.mutation_factor = np.clip(self.mutation_factor + adjustment, 0.5, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate + adjustment, 0.7, 1.0)
            self.learning_rate = np.clip(self.learning_rate + 0.01 * adjustment, 0.01, 0.1)

    def _resize_population(self, evaluations):
        if evaluations > self.budget * 0.25:
            self.population_size = max(4 * self.dim, self.init_population_size // 2)
        if evaluations > self.budget * 0.75:
            self.population_size = max(2 * self.dim, self.init_population_size // 4)

    def _update_covariance(self, population, centroid):
        deviations = population - centroid
        self.covariance_matrix = np.cov(deviations, rowvar=False)
        self.covariance_matrix += np.eye(self.dim) * 1e-6  # Regularization

    def _adaptive_learning(self, current_best_fitness):
        if self.history:
            recent_improvement = (self.history[-1] - current_best_fitness) / abs(self.history[-1])
            if recent_improvement < 0.01:
                self.learning_rate = np.clip(self.learning_rate * 1.1, 0.01, 0.1)
            else:
                self.learning_rate = np.clip(self.learning_rate * 0.9, 0.01, 0.1)
        self.history.append(current_best_fitness)

    def __call__(self, func):
        population = self._initialize_population()
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
            quantum_population = self._quantum_superposition(population, best_individual)
            orthogonal_population = self._orthogonal_exploration(quantum_population)
            combined_population = np.vstack((population, orthogonal_population))
            combined_fitness = np.apply_along_axis(func, 1, combined_population)
            evaluations += combined_population.shape[0]

            best_combined_idx = np.argmin(combined_fitness)
            if combined_fitness[best_combined_idx] < best_fitness:
                best_individual = combined_population[best_combined_idx]
                best_fitness = combined_fitness[best_combined_idx]

            self._resize_population(evaluations)
            self._adaptive_learning(best_fitness)

        return best_individual, best_fitness