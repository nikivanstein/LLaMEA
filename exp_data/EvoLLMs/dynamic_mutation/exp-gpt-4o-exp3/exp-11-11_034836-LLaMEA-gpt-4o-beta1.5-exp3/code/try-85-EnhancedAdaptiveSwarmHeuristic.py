import numpy as np

class EnhancedAdaptiveSwarmHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.init_population_size = 20 * dim
        self.population_size = self.init_population_size
        self.mutation_factor = 0.7
        self.crossover_rate = 0.9
        self.learning_rate = 0.1
        self.history = []
        self.best_positions = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.elite_fraction = 0.1

    def _initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = np.array([self.lb + (self.ub - self.lb) * np.random.random() for _ in range(self.dim)])
            population.append(individual)
        return np.array(population)

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

    def _adaptive_scaling(self, fitness):
        median_fitness = np.median(fitness)
        for idx, fit in enumerate(fitness):
            adjustment = self.learning_rate * (1 - 2 * (fit > median_fitness))
            self.mutation_factor = np.clip(self.mutation_factor + adjustment, 0.5, 1.0)

    def _elite_preservation(self, population, fitness):
        elite_size = max(1, int(self.elite_fraction * self.population_size))
        elite_indices = np.argsort(fitness)[:elite_size]
        elite_population = population[elite_indices]
        return elite_population

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.population_size

        while evaluations < self.budget:
            self._adaptive_scaling(fitness)
            elite_population = self._elite_preservation(population, fitness)
            
            for i in range(self.population_size):
                mutant = self._mutate(i, population)
                trial = self._crossover(population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.global_best_fitness:
                        self.global_best_fitness = trial_fitness
                        self.global_best_position = trial

                if evaluations >= self.budget:
                    break

            centroid = np.mean(elite_population, axis=0)
            levy_candidate = self._levy_flight(self.global_best_position, centroid)
            levy_fitness = func(levy_candidate)
            evaluations += 1

            if levy_fitness < self.global_best_fitness:
                self.global_best_fitness = levy_fitness
                self.global_best_position = levy_candidate

        return self.global_best_position, self.global_best_fitness