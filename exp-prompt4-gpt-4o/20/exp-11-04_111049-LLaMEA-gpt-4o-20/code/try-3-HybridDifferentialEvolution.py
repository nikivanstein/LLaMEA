import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.8
        self.CR = 0.9
        self.local_search_prob = 0.3
        self.adaptive_factor = 0.1  # New adaptive factor for tuning
        self.restart_threshold = 0.1  # Threshold for stagnation to trigger a restart

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _mutate(self, pop, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = pop[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _local_search(self, individual):
        direction = np.random.normal(0, 1, self.dim)
        step_size = np.random.uniform(0.01, 0.1)
        new_individual = np.clip(individual + step_size * direction, self.lower_bound, self.upper_bound)
        return new_individual

    def _adaptive_parameters(self):  # New function for adaptive parameter tuning
        self.F = np.clip(self.F + self.adaptive_factor * (np.random.rand() - 0.5), 0.5, 1.0)
        self.CR = np.clip(self.CR + self.adaptive_factor * (np.random.rand() - 0.5), 0.1, 1.0)

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size
        best_fitness = np.min(fitness)
        stagnation_counter = 0

        while evals < self.budget:
            for i in range(self.population_size):
                mutant = self._mutate(population, i)
                trial = self._crossover(population[i], mutant)
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    stagnation_counter = 0

                if evals >= self.budget:
                    break

                if np.random.rand() < self.local_search_prob:
                    local_candidate = self._local_search(population[i])
                    local_fitness = func(local_candidate)
                    evals += 1
                    if local_fitness < fitness[i]:
                        population[i] = local_candidate
                        fitness[i] = local_fitness
                        stagnation_counter = 0

                if evals >= self.budget:
                    break

            if np.min(fitness) >= best_fitness:
                stagnation_counter += 1
            else:
                best_fitness = np.min(fitness)
                stagnation_counter = 0

            if stagnation_counter > self.population_size * self.restart_threshold:
                population = self._initialize_population()
                fitness = np.array([func(ind) for ind in population])
                evals += self.population_size
                stagnation_counter = 0

            self._adaptive_parameters()

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]