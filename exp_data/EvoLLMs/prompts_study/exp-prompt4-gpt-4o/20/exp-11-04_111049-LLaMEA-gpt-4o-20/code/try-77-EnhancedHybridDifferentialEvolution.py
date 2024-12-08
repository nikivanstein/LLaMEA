import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim
        self.F = 0.5
        self.CR = 0.85
        self.local_search_prob = 0.4
        self.adaptive_alpha = 0.5
        self.performance_threshold = 0.1  # New: Performance metric threshold

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _mutate(self, pop, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = pop[np.random.choice(indices, 3, replace=False)]
        self.F = self.adaptive_alpha * (1.0 - np.random.rand() / 2)  
        mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        # New: Adaptive crossover strategy
        self.CR = 0.9 if np.std(target) > self.performance_threshold else 0.6 
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _local_search(self, individual, scale=1.0):
        direction = np.random.normal(0, 1, self.dim)
        step_size = np.random.uniform(0.05, 0.15) * scale
        new_individual = np.clip(individual + step_size * direction, self.lower_bound, self.upper_bound)
        return new_individual

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            for i in range(self.population_size):
                self.F = self.adaptive_alpha * (1.0 - evals / self.budget)
                mutant = self._mutate(population, i)
                trial = self._crossover(population[i], mutant)
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if evals >= self.budget:
                    break

                if np.random.rand() < self.local_search_prob:
                    scale_factor = 1 - (evals / self.budget)
                    local_candidate = self._local_search(population[i], scale=scale_factor)
                    local_fitness = func(local_candidate)
                    evals += 1
                    if local_fitness < fitness[i]:
                        population[i] = local_candidate
                        fitness[i] = local_fitness

                if evals >= self.budget:
                    break

            # New: Dynamic population resizing
            if evals < self.budget / 2:
                self.population_size = max(4, int(self.population_size * 0.95))

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]