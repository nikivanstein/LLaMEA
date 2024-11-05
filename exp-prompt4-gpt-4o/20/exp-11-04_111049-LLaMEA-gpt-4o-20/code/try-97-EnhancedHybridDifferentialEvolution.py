import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim
        self.CR = 0.85
        self.local_search_prob = 0.4
        self.adaptive_alpha = 0.5
        self.diversity_threshold = 0.1

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _mutate(self, pop, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = pop[np.random.choice(indices, 3, replace=False)]
        self.F = 0.1 + 0.9 * np.random.rand()
        mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant, rank):
        self.CR = 0.5 + 0.5 * (rank / self.population_size)
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _local_search(self, individual, scale=1.0, momentum=np.zeros((1,))):
        direction = np.random.normal(0, 1, self.dim)
        step_size = np.random.uniform(0.05, 0.15) * scale
        momentum = 0.7 * momentum + step_size * direction
        new_individual = np.clip(individual + momentum, self.lower_bound, self.upper_bound)
        return new_individual, momentum

    def _adaptive_scaling(self, diversity):
        return 0.1 + 0.9 * (1 - diversity / self.diversity_threshold)

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size
        momentum = np.zeros((self.population_size, self.dim))

        while evals < self.budget:
            diversity = np.std(population, axis=0).mean()
            if diversity < self.diversity_threshold:
                self.population_size = max(4, int(self.population_size * 0.9))
                population = population[:self.population_size]
                fitness = fitness[:self.population_size]

            for i in range(self.population_size):
                rank = np.argsort(fitness).tolist().index(i)
                mutant = self._mutate(population, i)
                trial = self._crossover(population[i], mutant, rank)
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if evals >= self.budget:
                    break

                if np.random.rand() < self.local_search_prob:
                    scale_factor = self._adaptive_scaling(diversity)
                    neighbors = np.random.choice(self.population_size, 5, replace=False)
                    best_neighbor = min(neighbors, key=lambda idx: fitness[idx])
                    direction = population[best_neighbor] - population[i]
                    local_candidate = np.clip(population[i] + 0.1 * direction, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_candidate)
                    evals += 1
                    if local_fitness < fitness[i]:
                        population[i] = local_candidate
                        fitness[i] = local_fitness

                if evals >= self.budget:
                    break

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]