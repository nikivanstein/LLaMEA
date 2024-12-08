import numpy as np

class AdaptiveDELS:
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

    def _initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

    def _mutate(self, target_idx, population, fitness):
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

    def _local_search(self, individual, best_individual):
        step_size = 0.05 + np.random.rand() * 0.1
        local_step = np.random.uniform(-step_size, step_size, self.dim)
        candidate = individual + local_step * (best_individual - individual)
        return np.clip(candidate, self.lb, self.ub)

    def _adapt_parameters(self, fitness, population):
        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(fitness)
        for idx in range(self.population_size):
            if fitness[idx] > fitness[best_idx]:
                self.mutation_factor = np.clip(self.mutation_factor + np.random.uniform(-self.adaptive_factor, self.adaptive_factor), 0.5, 1.0)
                self.crossover_rate = np.clip(self.crossover_rate + np.random.uniform(-self.adaptive_factor, self.adaptive_factor), 0.7, 1.0)
            else:
                self.mutation_factor = np.clip(self.mutation_factor + np.random.uniform(-self.adaptive_factor/2, self.adaptive_factor/2), 0.5, 1.0)
                self.crossover_rate = np.clip(self.crossover_rate + np.random.uniform(-self.adaptive_factor/2, self.adaptive_factor/2), 0.7, 1.0)

    def _resize_population(self, evaluations):
        if evaluations > self.budget * 0.75:
            self.population_size = max(4 * self.dim, self.population_size // 2)

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                self._adapt_parameters(fitness, population)
                mutant = self._mutate(i, population, fitness)
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

            local_candidate = self._local_search(best_individual, best_individual)
            local_fitness = func(local_candidate)
            evaluations += 1

            if local_fitness < best_fitness:
                best_individual = local_candidate
                best_fitness = local_fitness

            self._resize_population(evaluations)

        return best_individual, best_fitness